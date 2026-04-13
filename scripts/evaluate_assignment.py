import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Experiment:
    label: str
    model_path: Path
    data_yaml: Path
    imgsz: int
    conf: float
    split: str
    image_dir: Path


def load_images(folder: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in supported)


def measure_latency(model: YOLO, images: list[Path], imgsz: int, conf: float) -> tuple[float, float]:
    latencies: list[float] = []
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        started = time.perf_counter()
        model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        latencies.append((time.perf_counter() - started) * 1000)

    if not latencies:
        raise ValueError("No readable images found for latency measurement.")

    avg_latency = statistics.mean(latencies)
    if len(latencies) == 1:
        p95_latency = latencies[0]
    else:
        p95_latency = statistics.quantiles(latencies, n=20)[18]
    return round(avg_latency, 2), round(p95_latency, 2)


def evaluate_experiment(experiment: Experiment) -> dict:
    model = YOLO(str(experiment.model_path))
    metrics = model.val(data=str(experiment.data_yaml), split=experiment.split, imgsz=experiment.imgsz, verbose=False)
    images = load_images(experiment.image_dir)
    avg_latency_ms, p95_latency_ms = measure_latency(model, images, experiment.imgsz, experiment.conf)

    return {
        "experiment": experiment.label,
        "model_path": str(experiment.model_path),
        "split": experiment.split,
        "images_evaluated": len(images),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "fps": round(1000.0 / avg_latency_ms, 2) if avg_latency_ms else 0.0,
    }


def parse_experiment(raw: str) -> Experiment:
    parts = raw.split("|")
    if len(parts) != 7:
        raise ValueError(
            "Each --experiment must be formatted as "
            "'label|model_path|data_yaml|imgsz|conf|split|image_dir'"
        )
    label, model_path, data_yaml, imgsz, conf, split, image_dir = parts
    return Experiment(
        label=label,
        model_path=Path(model_path),
        data_yaml=Path(data_yaml),
        imgsz=int(imgsz),
        conf=float(conf),
        split=split,
        image_dir=Path(image_dir),
    )


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="label|model_path|data_yaml|imgsz|conf|split|image_dir",
    )
    parser.add_argument(
        "--output-json",
        default=PROJECT_ROOT / "results" / "assignment_results.json",
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        default=PROJECT_ROOT / "results" / "assignment_results.csv",
        type=Path,
    )
    args = parser.parse_args()

    experiments = [parse_experiment(item) for item in args.experiment]
    rows = [evaluate_experiment(experiment) for experiment in experiments]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2))
    write_csv(rows, args.output_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
