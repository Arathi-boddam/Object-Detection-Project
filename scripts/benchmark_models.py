import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_images(folder: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in supported)


def benchmark(model_path: Path, images: list[Path], imgsz: int, conf: float) -> dict:
    model = YOLO(str(model_path))
    latencies = []
    detections = []
    for image_path in images:
        frame = cv2.imread(str(image_path))
        started = time.perf_counter()
        result = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        detections.append(len(result.boxes))
    return {
        "model_path": str(model_path),
        "images": len(images),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 1 else round(latencies[0], 2),
        "avg_detections_per_image": round(statistics.mean(detections), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--output", default=PROJECT_ROOT / "benchmark_results.json", type=Path)
    args = parser.parse_args()

    images = load_images(args.images)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    report = benchmark(args.model, images, args.imgsz, args.conf)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
