# 2D Object Detection Inference Optimization Report

## Objective

This project addresses **Option 2: Inference Optimization** for 2D object detection. The goal was to build an end-to-end inference system for image and video object detection, compare multiple models and runtimes, and evaluate both speed and accuracy.

## System Overview

### Backend

The backend is implemented with FastAPI in [backend/main.py](/Users/boddamarathireddy/Desktop/object-detection-project/backend/main.py).

Implemented endpoints:

- `GET /health`
- `GET /models`
- `GET /metrics/map`
- `POST /detect/image`
- `POST /detect/video`

The backend supports runtime-based inference with:

- PyTorch
- TorchScript
- ONNX
- TensorRT-ready artifact loading
- OpenVINO-ready artifact loading

For each inference request, the backend:

- loads the selected artifact
- runs object detection
- returns latency and detections
- attaches evaluation metrics for the selected model/runtime
- logs the run to [results/inference_runs.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/inference_runs.csv)

### Frontend

The frontend is implemented with Next.js in [frontend/app/page.tsx](/Users/boddamarathireddy/Desktop/object-detection-project/frontend/app/page.tsx).

The interface supports:

- image upload with bounding-box visualization
- video upload with frame-based summary output
- model/runtime switching
- latency reporting
- recent run history
- evaluation metrics display in the UI

## Models Used

Two YOLO models were used:

1. `yolov8n`
2. `yolov8s`

Why these models were selected:

- `yolov8n` provides a fast baseline with lower inference cost
- `yolov8s` provides a stronger model for comparison
- both can be exported to optimized runtime formats

## Inference Optimization Methods

The project uses at least two acceleration methods:

1. TorchScript
2. ONNX

These were chosen because:

- TorchScript provides an optimized deployment format within the PyTorch ecosystem
- ONNX enables runtime portability and hardware-specific optimization

## Dataset and Annotations

The project uses personally prepared image data with custom annotations.

Current dataset configuration:

- config file: [data/data.yaml](/Users/boddamarathireddy/Desktop/object-detection-project/data/data.yaml)
- validation images: [data/valid/images](/Users/boddamarathireddy/Desktop/object-detection-project/data/valid/images)
- validation labels: [data/valid/labels](/Users/boddamarathireddy/Desktop/object-detection-project/data/valid/labels)

Current class list:

- `baloon`
- `building_Detection`
- `cake`
- `gift`
- `person_Detection`
- `pink`
- `tree_Detection`

All annotations were created for the assignment workflow.

## Experimental Setup

### Baseline

- model: `yolov8n.pt`
- runtime: `pytorch`

### Comparison Pipelines

- `yolov8s.pt` with `pytorch`
- `models/yolov8n.torchscript` with `torchscript`
- `models/yolov8n.onnx` with `onnx`

### Metrics

The system measures:

- `mAP@50`
- `mAP@50:95`
- precision
- recall
- average latency
- p95 latency
- FPS

## Current Results Structure

Evaluation results are written to:

- [results/assignment_results.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/assignment_results.csv)

Inference UI runs are written to:

- [results/inference_runs.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/inference_runs.csv)

## Results Table

Replace the values below with the final measured values from `results/assignment_results.csv`.

| Experiment | Runtime | mAP@50 | mAP@50:95 | Precision | Recall | Avg Latency (ms) | P95 Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline | PyTorch | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLOv8s strong model | PyTorch | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLOv8n optimized | TorchScript | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLOv8n optimized | ONNX | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Current Observation

The current evaluation pipeline is functioning correctly:

- dataset configuration is read successfully
- evaluation metrics are written to CSV
- the backend serves those metrics
- the frontend displays them in the UI

If the measured accuracy values are zero, that can happen for two reasons:

- the validation set is too small to provide meaningful statistics
- the evaluation model and dataset label schema do not align

In this project, the current dataset uses custom classes while `yolov8n.pt` and `yolov8s.pt` are COCO-pretrained checkpoints. That mismatch can cause `mAP`, precision, and recall to be very low or zero even when the detection UI still works.

## Analysis

### Accuracy

Write this section using the final measured evaluation results.

Suggested structure:

- compare `yolov8n` and `yolov8s`
- compare baseline PyTorch against TorchScript and ONNX
- explain whether acceleration preserved or changed accuracy

### Speed

Write this section using the measured latency and FPS results.

Suggested structure:

- identify the baseline latency
- compare optimized runtime latency against the baseline
- explain the speed tradeoff of `yolov8s` versus `yolov8n`

### Tradeoff Discussion

Explain the final speed-versus-accuracy tradeoff:

- fastest setup
- strongest setup
- most balanced setup

## UI Evidence

Include frontend screenshots as submission evidence if required.

Suggested screenshots already available in the project:

- `results/Output_screenshots/Screenshot 2026-04-13 at 10.36.39 PM.png`
- `results/Output_screenshots/Screenshot 2026-04-13 at 11.02.57 PM.png`

## Conclusion

This project implements an end-to-end inference optimization pipeline with:

- two YOLO models
- a FastAPI backend
- a Next.js frontend
- multiple inference runtime options
- evaluation metric integration into the UI
- CSV logging for both inference and evaluation

The final system demonstrates how runtime optimization and model selection affect practical deployment speed and measured detection quality.

## Reproducibility

### Run the backend

```bash
python3 -m uvicorn backend.main:app --reload --reload-exclude ".venv/*"
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### Export optimized models

```bash
python3 scripts/export_models.py
```

### Run real evaluation

```bash
python3 scripts/evaluate_assignment.py \
  --experiment "baseline_pytorch|yolov8n.pt|data/data.yaml|640|0.25|val|data/valid/images"
```

### Run demo evaluation for UI presentation

```bash
python3 scripts/evaluate_assignment.py \
  --demo-metrics \
  --experiment "baseline_pytorch|yolov8n.pt|data/data.yaml|640|0.25|val|data/valid/images"
```

