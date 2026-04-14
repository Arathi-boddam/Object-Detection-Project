# Object Detection Inference Optimization Project

This repository implements an end-to-end inference optimization workflow for 2D object detection.

The project includes:

- a FastAPI backend for image and video object detection
- a Next.js frontend for upload, visualization, and runtime comparison
- support for multiple model runtimes such as `pytorch`, `torchscript`, and `onnx`
- scripts for export, benchmarking, and assignment evaluation
- CSV logging for both inference runs and evaluation metrics

## Current Models and Runtimes

Primary models:

- `yolov8n`
- `yolov8s`

Supported runtimes in the codebase:

- `pytorch`
- `torchscript`
- `onnx`
- `tensorrt`
- `openvino`

Current comparison setup:

- baseline: `yolov8n` with `pytorch`
- strong model: `yolov8s` with `pytorch`
- optimization 1: `yolov8n` with `torchscript`
- optimization 2: `yolov8n` with `onnx`

## System Architecture

### Backend

The FastAPI backend is implemented in [backend/main.py](/Users/boddamarathireddy/Desktop/object-detection-project/backend/main.py).

Available endpoints:

- `GET /health`
- `GET /models`
- `GET /metrics/map`
- `POST /detect/image`
- `POST /detect/video`

The backend:

- loads the requested model artifact
- performs image or video inference
- returns detections and latency
- attaches evaluation metrics for the selected model/runtime
- logs every run to [results/inference_runs.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/inference_runs.csv)

### Frontend

The frontend is implemented in [frontend/app/page.tsx](/Users/boddamarathireddy/Desktop/object-detection-project/frontend/app/page.tsx).

The UI supports:

- image upload with bounding-box overlays
- video upload with frame-based summary metrics
- model/runtime selection
- latency and detection summaries
- dataset evaluation metrics display
- recent run history

## Dataset Layout

The current dataset config is [data/data.yaml](/Users/boddamarathireddy/Desktop/object-detection-project/data/data.yaml).

The current evaluation split uses:

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

## Setup

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python3 -m uvicorn backend.main:app --reload --reload-exclude ".venv/*"
```

Backend URLs:

- API docs: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/health`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

- `http://localhost:3000`

If needed:

```bash
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Export Optimized Models

Put the PyTorch checkpoints in the project root first:

- [yolov8n.pt](/Users/boddamarathireddy/Desktop/object-detection-project/yolov8n.pt)
- [yolov8s.pt](/Users/boddamarathireddy/Desktop/object-detection-project/yolov8s.pt)

Then run:

```bash
python3 scripts/export_models.py
```

This generates artifacts under [models](/Users/boddamarathireddy/Desktop/object-detection-project/models).

## Run Evaluation

Use this command to generate evaluation metrics from the annotated validation set:

```bash
python3 scripts/evaluate_assignment.py \
  --experiment "baseline_pytorch|yolov8n.pt|data/data.yaml|640|0.25|val|data/valid/images"
```

Evaluation outputs:

- [results/assignment_results.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/assignment_results.csv)
- [results/assignment_results.json](/Users/boddamarathireddy/Desktop/object-detection-project/results/assignment_results.json)

## Inference Logging

Every image or video run is appended to:

- [results/inference_runs.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/inference_runs.csv)

That file stores:

- media name and type
- model and runtime
- artifact
- latency
- detection counts
- video frame counts and duration
- top detections
- matched evaluation metrics

## Current Results

Current values from [results/inference_results.csv](/Users/boddamarathireddy/Desktop/object-detection-project/results/inference_results.csv):

| Experiment | Split | Images | mAP@50 | mAP@50:95 | Precision | Recall | Avg Latency (ms) | P95 Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_pytorch` | `val` | 30 | 72.82% | 82.70% | 92.59% | 80.91% | 84.28 | 158.73 | 11.87 |

## Known Limitation

If you evaluate a raw COCO-pretrained model such as `yolov8n.pt` directly against a custom dataset with non-COCO class definitions, accuracy can be very low or zero. That is expected. Meaningful `mAP` requires either:

- class definitions aligned with the pretrained model, or
- a custom model trained or fine-tuned on your annotated classes

## UI Screenshots

Main dashboard:

![Dashboard UI](results/Output_screenshots/Screenshot%202026-04-13%20at%2010.36.39%E2%80%AFPM.png)

Evaluation and run metrics:

![Metrics UI](results/Output_screenshots/Screenshot%202026-04-13%20at%2011.02.57%E2%80%AFPM.png)

## Important Files

- [backend/main.py](/Users/boddamarathireddy/Desktop/object-detection-project/backend/main.py)
- [frontend/app/page.tsx](/Users/boddamarathireddy/Desktop/object-detection-project/frontend/app/page.tsx)
- [frontend/app/globals.css](/Users/boddamarathireddy/Desktop/object-detection-project/frontend/app/globals.css)
- [scripts/export_models.py](/Users/boddamarathireddy/Desktop/object-detection-project/scripts/export_models.py)
- [scripts/benchmark_models.py](/Users/boddamarathireddy/Desktop/object-detection-project/scripts/benchmark_models.py)
- [scripts/evaluate_assignment.py](/Users/boddamarathireddy/Desktop/object-detection-project/scripts/evaluate_assignment.py)
- [REPORT.md](/Users/boddamarathireddy/Desktop/object-detection-project/REPORT.md)
