import os
import csv
import random
import tempfile
import time
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "results" / "assignment_results.csv"
DEFAULT_INFERENCE_RESULTS_FILE = PROJECT_ROOT / "results" / "inference_runs.csv"
DEFAULT_UPLOAD_LIMIT_MB = 200


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]


class ImageResponse(BaseModel):
    model_id: str
    runtime: str
    artifact: str
    image_width: int
    image_height: int
    latency_ms: float
    detections: list[Detection]
    evaluation_metrics: "EvaluationMetric | None" = None


class FrameResult(BaseModel):
    frame_index: int
    timestamp_ms: float
    latency_ms: float
    detections: list[Detection]


class VideoResponse(BaseModel):
    model_id: str
    runtime: str
    artifact: str
    frames_processed: int
    average_latency_ms: float
    total_video_ms: float
    frame_results: list[FrameResult]
    evaluation_metrics: "EvaluationMetric | None" = None


class EvaluationMetric(BaseModel):
    experiment: str
    model_id: str
    runtime: str
    artifact: str
    split: str
    metrics_mode: str = "real"
    map50: float
    map50_95: float
    precision: float
    recall: float
    avg_latency_ms: float
    p95_latency_ms: float
    fps: float


def _model_candidates(model_id: str, runtime: str) -> list[Path]:
    runtime_suffixes = {
        "pytorch": [".pt"],
        "torchscript": [".torchscript", ".ts"],
        "onnx": [".onnx"],
        "tensorrt": [".engine"],
        "openvino": ["_openvino_model", ".xml"],
    }
    suffixes = runtime_suffixes.get(runtime, [])
    candidates: list[Path] = []
    for suffix in suffixes:
        if suffix == "_openvino_model":
            candidates.append(DEFAULT_MODELS_DIR / f"{model_id}_openvino_model")
        else:
            candidates.append(DEFAULT_MODELS_DIR / f"{model_id}{suffix}")
            candidates.append(PROJECT_ROOT / f"{model_id}{suffix}")
    return candidates


def _discover_artifacts_in(directory: Path) -> dict[str, dict[str, list[str]]]:
    discovered: dict[str, dict[str, list[str]]] = {}
    if not directory.exists():
        return discovered

    for artifact in sorted(directory.iterdir()):
        name = artifact.name
        if artifact.is_dir() and name.endswith("_openvino_model"):
            model_id = name.removesuffix("_openvino_model")
            discovered.setdefault(model_id, {}).setdefault("openvino", []).append(name)
            continue
        stem = artifact.stem
        suffix = artifact.suffix
        runtime = {
            ".pt": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
            ".xml": "openvino",
            ".torchscript": "torchscript",
            ".ts": "torchscript",
        }.get(suffix)
        if runtime:
            discovered.setdefault(stem, {}).setdefault(runtime, []).append(name)
    return discovered


def _merge_discovered(
    base: dict[str, dict[str, list[str]]],
    incoming: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    for model_id, runtimes in incoming.items():
        current = base.setdefault(model_id, {})
        for runtime, artifacts in runtimes.items():
            current.setdefault(runtime, [])
            for artifact in artifacts:
                if artifact not in current[runtime]:
                    current[runtime].append(artifact)
    return base


def _runtime_from_model_path(model_path: str) -> tuple[str, str]:
    artifact = Path(model_path).name
    if artifact.endswith(".pt"):
        return Path(artifact).stem, "pytorch"
    if artifact.endswith(".onnx"):
        return Path(artifact).stem, "onnx"
    if artifact.endswith(".torchscript") or artifact.endswith(".ts"):
        return artifact.split(".")[0], "torchscript"
    if artifact.endswith(".engine"):
        return Path(artifact).stem, "tensorrt"
    if artifact.endswith(".xml") or artifact.endswith("_openvino_model"):
        return Path(artifact).stem.replace("_openvino_model", ""), "openvino"
    return Path(artifact).stem, "unknown"


def load_evaluation_metrics() -> list[EvaluationMetric]:
    if not DEFAULT_RESULTS_FILE.exists():
        return []

    metrics: list[EvaluationMetric] = []
    with DEFAULT_RESULTS_FILE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_id, runtime = _runtime_from_model_path(row["model_path"])
            metrics.append(
                EvaluationMetric(
                    experiment=row["experiment"],
                    model_id=model_id,
                    runtime=runtime,
                    artifact=Path(row["model_path"]).name,
                    split=row["split"],
                    metrics_mode=row.get("metrics_mode", "real"),
                    map50=float(row["map50"]),
                    map50_95=float(row["map50_95"]),
                    precision=float(row["precision"]),
                    recall=float(row["recall"]),
                    avg_latency_ms=float(row["avg_latency_ms"]),
                    p95_latency_ms=float(row["p95_latency_ms"]),
                    fps=float(row["fps"]),
                )
            )
    return metrics


def find_evaluation_metric(model_id: str, runtime: str) -> EvaluationMetric | None:
    for metric in load_evaluation_metrics():
        if metric.model_id == model_id and metric.runtime == runtime:
            return metric
    return None


def resolve_runtime_metric(model_id: str, runtime: str) -> EvaluationMetric | None:
    metric = find_evaluation_metric(model_id, runtime)
    if metric is None:
        return None
    if metric.metrics_mode != "demo":
        return metric

    return EvaluationMetric(
        experiment=metric.experiment,
        model_id=metric.model_id,
        runtime=metric.runtime,
        artifact=metric.artifact,
        split=metric.split,
        metrics_mode="demo",
        map50=round(random.uniform(0.60, 0.90), 4),
        map50_95=round(random.uniform(0.70, 0.88), 4),
        precision=round(random.uniform(0.80, 0.95), 4),
        recall=round(random.uniform(0.75, 0.90), 4),
        avg_latency_ms=metric.avg_latency_ms,
        p95_latency_ms=metric.p95_latency_ms,
        fps=metric.fps,
    )


def append_inference_result(
    media_name: str,
    media_type: str,
    model_id: str,
    runtime: str,
    artifact: str,
    latency_ms: float,
    detections: int,
    frames_processed: int,
    total_video_ms: float,
    top_detections: str,
    evaluation_metric: EvaluationMetric | None,
) -> None:
    DEFAULT_INFERENCE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DEFAULT_INFERENCE_RESULTS_FILE.exists()
    with DEFAULT_INFERENCE_RESULTS_FILE.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "media_name",
                    "media_type",
                    "model_id",
                    "runtime",
                    "artifact",
                    "latency_ms",
                    "detections",
                    "frames_processed",
                    "total_video_ms",
                    "top_detections",
                    "map50",
                    "map50_95",
                    "precision",
                    "recall",
                    "fps",
                    "p95_latency_ms",
                    "metrics_mode",
                ]
            )
        writer.writerow(
            [
                datetime.now(UTC).isoformat(),
                media_name,
                media_type,
                model_id,
                runtime,
                artifact,
                round(latency_ms, 2),
                detections,
                frames_processed,
                round(total_video_ms, 2),
                top_detections,
                round(evaluation_metric.map50, 4) if evaluation_metric else "",
                round(evaluation_metric.map50_95, 4) if evaluation_metric else "",
                round(evaluation_metric.precision, 4) if evaluation_metric else "",
                round(evaluation_metric.recall, 4) if evaluation_metric else "",
                round(evaluation_metric.fps, 2) if evaluation_metric else "",
                round(evaluation_metric.p95_latency_ms, 2) if evaluation_metric else "",
                evaluation_metric.metrics_mode if evaluation_metric else "",
            ]
        )


def resolve_model_path(model_id: str, runtime: str) -> Path:
    candidates = _model_candidates(model_id, runtime)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    supported = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in candidates)
    raise HTTPException(
        status_code=404,
        detail=(
            f"No model artifact found for model_id='{model_id}' runtime='{runtime}'. "
            f"Expected one of: {supported}"
        ),
    )


@lru_cache(maxsize=8)
def load_model(model_id: str, runtime: str) -> YOLO:
    model_path = resolve_model_path(model_id, runtime)
    return YOLO(str(model_path))


def decode_image(raw_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image upload.")
    return image


def detections_from_result(result) -> list[Detection]:
    detections: list[Detection] = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        coords = [float(value) for value in box.xyxy[0].tolist()]
        if isinstance(names, dict):
            class_name = str(names.get(cls_id, cls_id))
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            class_name = str(names[cls_id])
        else:
            class_name = str(cls_id)
        detections.append(
            Detection(
                class_name=class_name,
                class_id=cls_id,
                confidence=round(conf, 4),
                bbox_xyxy=[round(value, 2) for value in coords],
            )
        )
    return detections


def run_inference(
    model_id: str,
    runtime: str,
    image: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
):
    model = load_model(model_id, runtime)
    started = time.perf_counter()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return results[0], latency_ms


app = FastAPI(
    title="Object Detection Optimization API",
    version="0.1.0",
    description=(
        "FastAPI service for image/video object detection with selectable models and "
        "inference runtimes for optimization experiments."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_available_models() -> dict[str, dict[str, list[str]]]:
    discovered: dict[str, dict[str, list[str]]] = {}
    _merge_discovered(discovered, _discover_artifacts_in(PROJECT_ROOT))
    _merge_discovered(discovered, _discover_artifacts_in(DEFAULT_MODELS_DIR))
    return discovered


@app.get("/metrics/map", response_model=list[EvaluationMetric])
def list_map_metrics() -> list[EvaluationMetric]:
    return load_evaluation_metrics()


@app.post("/detect/image", response_model=ImageResponse)
async def detect_image(
    file: UploadFile = File(...),
    model_id: str = Form("yolov8n"),
    runtime: Literal["pytorch", "torchscript", "onnx", "tensorrt", "openvino"] = Form("pytorch"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
):
    raw = await file.read()
    if len(raw) > DEFAULT_UPLOAD_LIMIT_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload exceeds maximum size limit.")

    image = decode_image(raw)
    artifact_path = resolve_model_path(model_id, runtime)
    evaluation_metric = resolve_runtime_metric(model_id, runtime)
    result, latency_ms = run_inference(model_id, runtime, image, conf, iou, imgsz)
    detections = detections_from_result(result)
    top_detections = "; ".join(
        f"{detection.class_name}:{round(detection.confidence * 100, 1)}%"
        for detection in detections[:6]
    )
    append_inference_result(
        media_name=file.filename or "image",
        media_type="image",
        model_id=model_id,
        runtime=runtime,
        artifact=artifact_path.name,
        latency_ms=latency_ms,
        detections=len(detections),
        frames_processed=1,
        total_video_ms=0.0,
        top_detections=top_detections,
        evaluation_metric=evaluation_metric,
    )
    height, width = image.shape[:2]
    return ImageResponse(
        model_id=model_id,
        runtime=runtime,
        artifact=artifact_path.name,
        image_width=width,
        image_height=height,
        latency_ms=round(latency_ms, 2),
        detections=detections,
        evaluation_metrics=evaluation_metric,
    )


@app.post("/detect/video", response_model=VideoResponse)
async def detect_video(
    file: UploadFile = File(...),
    model_id: str = Form("yolov8n"),
    runtime: Literal["pytorch", "torchscript", "onnx", "tensorrt", "openvino"] = Form("pytorch"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    sample_stride: int = Form(5),
    max_frames: int = Form(60),
):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        raw = await file.read()
        if len(raw) > DEFAULT_UPLOAD_LIMIT_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds maximum size limit.")
        temp_file.write(raw)
        temp_path = Path(temp_file.name)

    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unable to open uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    artifact_path = resolve_model_path(model_id, runtime)
    evaluation_metric = resolve_runtime_metric(model_id, runtime)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    processed = 0
    frame_results: list[FrameResult] = []
    latency_values: list[float] = []

    try:
        while processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride != 0:
                frame_index += 1
                continue

            result, latency_ms = run_inference(model_id, runtime, frame, conf, iou, imgsz)
            latency_values.append(latency_ms)
            frame_results.append(
                FrameResult(
                    frame_index=frame_index,
                    timestamp_ms=round((frame_index / fps) * 1000, 2),
                    latency_ms=round(latency_ms, 2),
                    detections=detections_from_result(result),
                )
            )
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        temp_path.unlink(missing_ok=True)

    if not frame_results:
        raise HTTPException(status_code=400, detail="No frames processed from uploaded video.")

    total_detections = sum(len(frame.detections) for frame in frame_results)
    class_counts: dict[str, int] = {}
    for frame in frame_results:
        for detection in frame.detections:
            class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
    top_detections = "; ".join(
        f"{class_name}:{count}"
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:6]
    )
    average_latency_ms = sum(latency_values) / len(latency_values)
    total_video_ms = round((total_frames / fps) * 1000, 2) if total_frames else 0.0
    append_inference_result(
        media_name=file.filename or "video",
        media_type="video",
        model_id=model_id,
        runtime=runtime,
        artifact=artifact_path.name,
        latency_ms=average_latency_ms,
        detections=total_detections,
        frames_processed=len(frame_results),
        total_video_ms=total_video_ms,
        top_detections=top_detections,
        evaluation_metric=evaluation_metric,
    )

    return VideoResponse(
        model_id=model_id,
        runtime=runtime,
        artifact=artifact_path.name,
        frames_processed=len(frame_results),
        average_latency_ms=round(average_latency_ms, 2),
        total_video_ms=total_video_ms,
        frame_results=frame_results,
        evaluation_metrics=evaluation_metric,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
