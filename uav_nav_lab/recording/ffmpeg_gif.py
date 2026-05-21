"""PNG-sequence → GIF builder with optional frame decimation.

All recording scripts used the same two-pass palettegen + paletteuse
pipeline. The only meaningful variation was whether to decimate the
input sequence first:

* "decimated" style (single + multi demos, side-by-side compares):
  count the available PNGs, work out how many ``keep_every`` to drop
  so the GIF lands in ``target_seconds`` at ``fps``, then use
  ``select='not(mod(n,K))', setpts=N/fps/TB`` in the vf chain.
* "simple" style (top-down recorder): no decimation; the source was
  captured at the target fps, so ``fps={fps}`` is enough.

Set ``target_seconds=None`` to fall through to the simple variant.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def count_frames(frames_dir: Path, *, name_contains: str | None = None) -> int:
    """Count ``.png`` files in ``frames_dir``.

    When ``name_contains`` is given (e.g. ``"front_center"``), only
    frames whose filename matches are counted — the recording scripts
    save multi-camera frames into the same directory, so the count
    used for decimation needs to be over the relevant camera only.
    """
    n = 0
    for p in frames_dir.iterdir():
        if p.suffix != ".png":
            continue
        if name_contains is not None and name_contains not in p.name:
            continue
        n += 1
    return n


def build_ffmpeg_vf(
    *,
    fps: int,
    width: int,
    keep_every: int | None,
) -> str:
    """Build the ``-vf`` filter string for the two-pass GIF encode.

    ``keep_every=None`` (or ``1``) skips the decimation step entirely
    and uses the simple ``fps={fps}`` form. Otherwise the chain is
    ``select='not(mod(n,K))', setpts=N/{fps}/TB`` which keeps every
    K-th source frame and re-stamps timing so playback runs at
    ``fps`` regardless of the source rate.
    """
    scale = f"scale={width}:-1:flags=lanczos"
    if keep_every is None or keep_every <= 1:
        return f"fps={fps},{scale}"
    return (
        f"select='not(mod(n,{keep_every}))',"
        f"setpts=N/{fps}/TB,"
        f"{scale}"
    )


def frames_to_gif(
    frames_dir: Path,
    out: Path,
    *,
    fps: int = 15,
    width: int = 480,
    target_seconds: float | None = 7.0,
    frame_pattern: str = "step_%04d_front_center.png",
    name_contains: str | None = "front_center",
    quiet: bool = True,
) -> int:
    """Encode the PNG sequence under ``frames_dir`` into a GIF at ``out``.

    Returns the number of source frames seen (useful for caller log
    lines).

    * ``target_seconds`` — desired output GIF duration; when set, the
      function counts source frames and decimates so the encode lands
      in roughly that long at the chosen ``fps``. ``None`` skips
      decimation (top-down recorder).
    * ``frame_pattern`` — ffmpeg input pattern (printf-style); pass a
      different one (e.g. ``"frame_%04d.png"``) for non-step-prefixed
      captures.
    * ``name_contains`` — substring filter used by :func:`count_frames`
      when computing ``keep_every``. ``None`` counts every PNG.
    """
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"{frames_dir} not found")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_frames = count_frames(frames_dir, name_contains=name_contains)
    if target_seconds is None:
        keep_every: int | None = None
    else:
        desired = max(1, int(round(fps * target_seconds)))
        keep_every = max(1, n_frames // desired) if n_frames else 1

    vf = build_ffmpeg_vf(fps=fps, width=width, keep_every=keep_every)
    palette = frames_dir / "_palette.png"
    pattern = str(frames_dir / frame_pattern)

    sink = subprocess.DEVNULL if quiet else None
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern,
         "-vf", f"{vf},palettegen=stats_mode=diff",
         str(palette)],
        check=True, stdout=sink, stderr=sink,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-i", str(palette),
         "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
         "-loop", "0",
         str(out)],
        check=True, stdout=sink, stderr=sink,
    )
    palette.unlink(missing_ok=True)
    return n_frames
