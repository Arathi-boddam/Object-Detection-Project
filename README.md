# Object Detection Inference Optimization Project

This repository is set up for **Option 2: Inference Optimization**.

## What is included

- A FastAPI backend with:
  - `POST /detect/image`
  - `POST /detect/video`
  - model/runtime selection for `pytorch`, `torchscript`, `onnx`, `tensorrt`, `openvino`
- A Next.js frontend for:
  - uploading image/video
  - calling the backend
  - showing latency
  - drawing bounding boxes for image inference
- Scripts to:
  - export YOLO checkpoints into optimized runtime formats
  - benchmark latency over a folder of evaluation images
  - evaluate assignment experiments and generate result tables
- A submission-ready report draft in `REPORT.md`

## Recommended experiment design

Use at least two strong models, for example:

- `yolov8n.pt`
- `yolov8s.pt`

Use at least two acceleration methods, for example:

- `torchscript`
- `onnx`

If your machine supports them, you can also export and compare:

- `tensorrt`
- `openvino`

## Suggested folder usage

- `data/images/`: evaluation images from your own dataset
- `data/videos/`: evaluation videos from your own dataset
- `data/labels/`: your own annotations
- `models/`: exported runtime artifacts

## Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python3 -m uvicorn backend.main:app --reload
```

API docs will be available at `http://127.0.0.1:8000/docs`.

## Frontend setup

```bash
cd frontend
npm install
pip install -r backend/requirements.txt
```

Set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000` if needed.

## Export optimized models

Put your source checkpoints in the project root first, for example:

- `yolov8n.pt`
- `yolov8s.pt`

Then run:

```bash
python3 scripts/export_models.py
```

## Benchmark latency

```bash
python3 scripts/benchmark_models.py \
  --images data/images \
  --model models/yolov8n.onnx
```

## Run the full assignment evaluation

Add your own annotated dataset first, then run:

```bash
python3 scripts/evaluate_assignment.py \
  --experiment "baseline_pytorch|yolov8n.pt|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "strong_model_pytorch|yolov8s.pt|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "optimized_torchscript|models/yolov8n.torchscript|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "optimized_onnx|models/yolov8n.onnx|data/dataset.yaml|640|0.25|val|data/images/val"
```

This writes:

- `results/assignment_results.json`
- `results/assignment_results.csv`
- values you can paste directly into `REPORT.md`

## What to report for the assignment

Your final report should include:

1. Baseline model/runtime:
   - Example: `yolov8n.pt` with PyTorch inference
2. Improved inference pipelines:
   - Example: `yolov8n.onnx`
   - Example: `yolov8s.torchscript`
3. Accuracy comparison:
   - mAP on your own annotated data
4. Speed comparison:
   - average latency
   - p95 latency
   - FPS if you want to add it
5. Explanation:
   - why these models were chosen
   - why these acceleration methods were chosen
   - tradeoff between speed and accuracy

## Minimum submission story

An easy credible setup for the rubric is:

- Baseline: `yolov8n` with PyTorch
- Model 2: `yolov8s` with PyTorch
- Optimization 1: `yolov8n` with TorchScript
- Optimization 2: `yolov8n` with ONNX

That gives you:

- two strong-performing models
- a backend
- a frontend
- two acceleration methods
- measurable latency and accuracy comparisons

## Files you will submit

- source code in this repository
- your own annotated images/videos under `data/`
- generated model artifacts under `models/`
- completed results under `results/`
- completed report in `REPORT.md`
