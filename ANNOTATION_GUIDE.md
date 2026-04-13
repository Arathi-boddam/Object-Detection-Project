# Annotation Guide

This project needs **your own annotated images/videos** for valid accuracy and mAP evaluation.

## 1. Create your dataset folders

Use this structure:

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  videos/
    raw/
```

For a small class project, `val/` is enough to get started.

## 2. Add your own images or extract frames from your own video

If you already have your own photos/screenshots:

- put them in `data/images/val/`

If you have your own video:

```bash
source .venv/bin/activate
python3 scripts/extract_frames.py \
  --video data/videos/raw/my_video.mp4 \
  --output data/images/val \
  --every-n-frames 30 \
  --prefix picnic
```

## 3. Annotate them

You already have `label-studio` installed in the virtualenv.

Start it locally:

```bash
source .venv/bin/activate
label-studio start
```

Then:

1. Open the local Label Studio URL in your browser
2. Create a new project
3. Import images from `data/images/val/`
4. Add rectangle labels for your classes
5. Annotate every object you want to evaluate

Use classes that actually appear in your data, for example:

- `person`
- `bottle`
- `chair`
- `car`

## 4. Export annotations in YOLO format

After annotation, export your labels in **YOLO** format.

Place:

- images in `data/images/val/`
- label `.txt` files in `data/labels/val/`

Each image must have a matching label file with the same base name.

Example:

- `data/images/val/picnic_0001.jpg`
- `data/labels/val/picnic_0001.txt`

## 5. Update `data/dataset.yaml`

Set your class names in [data/dataset.yaml](/Users/boddamarathireddy/Desktop/object-detection-project/data/dataset.yaml).

Example:

```yaml
path: data
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: bottle
```

## 6. Run evaluation

```bash
source .venv/bin/activate
python3 scripts/evaluate_assignment.py \
  --experiment "baseline_pytorch|yolov8n.pt|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "strong_model_pytorch|yolov8s.pt|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "optimized_torchscript|models/yolov8n.torchscript|data/dataset.yaml|640|0.25|val|data/images/val" \
  --experiment "optimized_onnx|models/yolov8n.onnx|data/dataset.yaml|640|0.25|val|data/images/val"
```

That generates real:

- `mAP@50`
- `mAP@50:95`
- precision
- recall
- latency
- FPS

## Notes

- Do not use stock images if your instructor requires strictly your own data.
- Confidence scores from the frontend are not mAP.
- mAP only becomes valid after you annotate your own dataset.
