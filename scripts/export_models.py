from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def export_all(model_name: str) -> None:
    model_path = PROJECT_ROOT / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing source weights: {model_path}")

    model = YOLO(str(model_path))
    torchscript_path = Path(model.export(format="torchscript", imgsz=640))
    onnx_path = Path(model.export(format="onnx", imgsz=640, dynamic=True))

    target_torchscript = MODELS_DIR / torchscript_path.name
    target_onnx = MODELS_DIR / onnx_path.name

    if torchscript_path.resolve() != target_torchscript.resolve():
        torchscript_path.replace(target_torchscript)
    if onnx_path.resolve() != target_onnx.resolve():
        onnx_path.replace(target_onnx)


if __name__ == "__main__":
    for model_name in ("yolov8n", "yolov8s"):
        try:
            export_all(model_name)
            print(f"Exported runtime variants for {model_name}")
        except FileNotFoundError as error:
            print(error)
