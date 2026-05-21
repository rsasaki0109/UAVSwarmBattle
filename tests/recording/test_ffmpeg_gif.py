"""Characterization tests for uav_nav_lab.recording.ffmpeg_gif.

Pin the frame-counting filter, the vf-string assembly (with and
without decimation), and the two-pass ffmpeg invocation pattern via
a mocked subprocess.run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from uav_nav_lab.recording import build_ffmpeg_vf, count_frames, frames_to_gif


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic — content not parsed


def test_count_frames_all_pngs_when_no_filter(tmp_path: Path):
    _touch(tmp_path / "step_0001_front_center.png")
    _touch(tmp_path / "step_0002_front_center.png")
    _touch(tmp_path / "step_0001_back_center.png")
    _touch(tmp_path / "_palette.png")
    _touch(tmp_path / "notes.txt")  # non-PNG ignored
    assert count_frames(tmp_path) == 4


def test_count_frames_filters_by_substring(tmp_path: Path):
    _touch(tmp_path / "step_0001_front_center.png")
    _touch(tmp_path / "step_0002_front_center.png")
    _touch(tmp_path / "step_0001_back_center.png")  # excluded
    assert count_frames(tmp_path, name_contains="front_center") == 2


def test_count_frames_returns_zero_for_empty_dir(tmp_path: Path):
    assert count_frames(tmp_path) == 0


def test_build_ffmpeg_vf_simple_mode_when_keep_every_is_none():
    vf = build_ffmpeg_vf(fps=15, width=480, keep_every=None)
    assert vf == "fps=15,scale=480:-1:flags=lanczos"


def test_build_ffmpeg_vf_simple_mode_when_keep_every_is_one():
    # keep_every=1 means "keep every frame" — semantically same as no decimation
    vf = build_ffmpeg_vf(fps=15, width=480, keep_every=1)
    assert vf == "fps=15,scale=480:-1:flags=lanczos"


def test_build_ffmpeg_vf_decimation_chain_with_keep_every_three():
    vf = build_ffmpeg_vf(fps=12, width=320, keep_every=3)
    assert vf == (
        "select='not(mod(n,3))',"
        "setpts=N/12/TB,"
        "scale=320:-1:flags=lanczos"
    )


def test_frames_to_gif_runs_two_pass_ffmpeg_with_correct_keep_every(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # 100 source frames, fps=10, target_seconds=5.0 → desired=50 → keep_every=2
    frames_dir = tmp_path / "frames"
    for i in range(100):
        _touch(frames_dir / f"step_{i:04d}_front_center.png")
    out_gif = tmp_path / "out" / "demo.gif"

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool, stdout: Any = None, stderr: Any = None,
                 **kwargs: Any) -> Any:
        calls.append(cmd)
        # Simulate the encoder by touching the requested output path
        # (the palette + final GIF — last argv element is the output).
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"GIF")
        return None

    monkeypatch.setattr("uav_nav_lab.recording.ffmpeg_gif.subprocess.run", fake_run)

    n = frames_to_gif(
        frames_dir, out_gif,
        fps=10, width=480, target_seconds=5.0,
    )
    assert n == 100
    assert len(calls) == 2  # palettegen + paletteuse
    # The vf string used in both calls should reflect keep_every=2
    assert any("not(mod(n,2))" in arg for arg in calls[0])
    assert any("not(mod(n,2))" in arg for arg in calls[1])
    # Final encode targets the requested out path
    assert calls[1][-1] == str(out_gif)


def test_frames_to_gif_simple_mode_skips_decimation_when_target_seconds_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames_dir = tmp_path / "frames"
    for i in range(20):
        _touch(frames_dir / f"frame_{i:04d}.png")
    out_gif = tmp_path / "out" / "demo.gif"

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "uav_nav_lab.recording.ffmpeg_gif.subprocess.run",
        lambda cmd, **kwargs: (calls.append(cmd),
                               Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True),
                               Path(cmd[-1]).write_bytes(b"GIF"))[-1],
    )

    n = frames_to_gif(
        frames_dir, out_gif,
        fps=10, width=640,
        target_seconds=None,            # simple mode
        frame_pattern="frame_%04d.png",
        name_contains=None,
    )
    assert n == 20
    assert len(calls) == 2
    # Simple-mode vf — no select/mod call, just fps + scale
    assert all("not(mod" not in arg for arg in calls[0])
    assert all("not(mod" not in arg for arg in calls[1])


def test_frames_to_gif_raises_when_frames_dir_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        frames_to_gif(tmp_path / "nope", tmp_path / "out.gif")
