#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames as zero-padded PNG files (e.g. 000000.png)."
    )
    parser.add_argument("video", type=Path, help="Input video path, e.g. sushe.mp4")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: example/<video_name_without_suffix>",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target extraction FPS (must be > 0). Default: 1.0",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting frame index for output names. Default: 0",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Zero-padding width for output names. Default: 6",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.digits <= 0:
        raise ValueError("--digits must be > 0")

    output_dir = args.output_dir or Path("example") / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        raise RuntimeError("Cannot read source FPS from video.")

    # Step through source frames according to target FPS.
    frame_step = max(1, int(round(src_fps / args.fps)))

    src_idx = 0
    out_idx = args.start_index
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if src_idx % frame_step == 0:
            out_name = f"{out_idx:0{args.digits}d}.png"
            out_path = output_dir / out_name
            if not cv2.imwrite(str(out_path), frame):
                cap.release()
                raise RuntimeError(f"Failed to write frame: {out_path}")
            out_idx += 1
            saved += 1

        src_idx += 1

    cap.release()

    print(f"Done. Saved {saved} frames to: {output_dir}")
    print(f"Naming format example: {(output_dir / f'{args.start_index:0{args.digits}d}.png')}")


if __name__ == "__main__":
    main()
