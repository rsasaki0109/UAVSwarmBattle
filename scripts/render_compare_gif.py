#!/usr/bin/env python3
"""Stitch two episode GIFs side-by-side with a title label per pane.

Usage:
  scripts/render_compare_gif.py left.gif right.gif --out compare.gif \
      --left-label "GPU MPPI" --right-label "CPU MPC" --fps 15

Both inputs must already be rendered (use `uav-nav anim`). If their
frame counts differ, the shorter side is held on its last frame so
both panes finish together.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_frames(path: Path) -> list[Image.Image]:
    im = Image.open(path)
    frames: list[Image.Image] = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert("RGB").copy())
    return frames


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def compose(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    left_label: str,
    right_label: str,
    fps: int,
    header_px: int = 32,
    max_total_width: int | None = None,
    frame_stride: int = 1,
) -> None:
    left_frames = _load_frames(left_path)
    right_frames = _load_frames(right_path)
    if not left_frames or not right_frames:
        raise ValueError("one of the inputs has zero frames")

    if frame_stride > 1:
        left_frames = left_frames[::frame_stride]
        right_frames = right_frames[::frame_stride]

    target_h = max(left_frames[0].height, right_frames[0].height)

    def _scale(im: Image.Image) -> Image.Image:
        if im.height == target_h:
            return im
        w = int(round(im.width * target_h / im.height))
        return im.resize((w, target_h), Image.LANCZOS)

    left_frames = [_scale(f) for f in left_frames]
    right_frames = [_scale(f) for f in right_frames]

    combined_w = left_frames[0].width + right_frames[0].width
    if max_total_width is not None and combined_w > max_total_width:
        scale = max_total_width / combined_w
        new_h = int(round(target_h * scale))

        def _shrink(im: Image.Image) -> Image.Image:
            w = int(round(im.width * scale))
            return im.resize((w, new_h), Image.LANCZOS)

        left_frames = [_shrink(f) for f in left_frames]
        right_frames = [_shrink(f) for f in right_frames]
        target_h = new_h

    n = max(len(left_frames), len(right_frames))

    def _at(frames: list[Image.Image], i: int) -> Image.Image:
        return frames[i] if i < len(frames) else frames[-1]

    font = _font(size=max(14, header_px - 10))
    composed: list[Image.Image] = []
    for i in range(n):
        l = _at(left_frames, i)
        r = _at(right_frames, i)
        w = l.width + r.width
        h = target_h + header_px
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(l, (0, header_px))
        canvas.paste(r, (l.width, header_px))
        draw = ImageDraw.Draw(canvas)
        # Center each label over its pane.
        for text, x_center in [(left_label, l.width / 2), (right_label, l.width + r.width / 2)]:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x_center - tw / 2, (header_px - th) / 2 - 2),
                      text, font=font, fill=(0, 0, 0))
        composed.append(canvas)

    duration_ms = int(round(1000.0 / max(1, fps)))
    composed[0].save(
        out_path,
        save_all=True,
        append_images=composed[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-total-width", type=int, default=None,
                   help="downscale composed frames to fit this total width (px)")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="keep every Nth source frame (>=1)")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    compose(
        args.left, args.right, args.out,
        left_label=args.left_label, right_label=args.right_label,
        fps=args.fps,
        max_total_width=args.max_total_width,
        frame_stride=max(1, args.frame_stride),
    )
    print(f"[render_compare_gif] {args.out}  ({args.left.name} | {args.right.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
