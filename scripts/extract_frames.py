import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, every_n_frames: int, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    frame_index = 0
    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                filename = output_dir / f"{prefix}_{saved:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved += 1
            frame_index += 1
    finally:
        capture.release()

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from your own video for annotation.")
    parser.add_argument("--video", required=True, type=Path, help="Path to your input video")
    parser.add_argument(
        "--output",
        default=Path("data/images/val"),
        type=Path,
        help="Folder to save extracted frames",
    )
    parser.add_argument(
        "--every-n-frames",
        default=30,
        type=int,
        help="Save one frame every N frames",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        type=str,
        help="Filename prefix for saved frames",
    )
    args = parser.parse_args()

    saved = extract_frames(args.video, args.output, args.every_n_frames, args.prefix)
    print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
