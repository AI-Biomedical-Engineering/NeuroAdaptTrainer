"""
retrain_model.py

Retrains/adapts the YOLO neuron segmentation model using annotations
generated from the Fiji/ImageJ transfer learning module.

Usage:
    python retrain_model.py <data_yaml_path> <output_model_path> [start_model_path] [device]
"""

from pathlib import Path
import sys
import traceback
import shutil
from ultralytics import YOLO
import torch


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "best.pt"

DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 25
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 2


def resolve_device(device):
    if device is None:
        return None

    if device == "auto_acceleration":
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        print(
            "Hardware acceleration was requested, but CUDA/MPS is not available. Falling back to CPU.",
            flush=True
        )
        return "cpu"

    return device


def main():
    try:
        if len(sys.argv) < 3:
            print(
                "Usage: python retrain_model.py <data_yaml_path> <output_model_path> [start_model_path] [device]",
                flush=True
            )
            sys.exit(2)

        data_yaml = Path(sys.argv[1]).expanduser().resolve()
        output_model = Path(sys.argv[2]).expanduser().resolve()

        if len(sys.argv) >= 4:
            training_start_model = Path(sys.argv[3]).expanduser().resolve()
        else:
            training_start_model = BASE_MODEL

        if len(sys.argv) >= 5:
            device = sys.argv[4]
        else:
            device = None
            
        resolved_device = resolve_device(device)

        if not data_yaml.exists():
            print(f"data.yaml not found: {data_yaml}", flush=True)
            sys.exit(2)

        if not training_start_model.exists():
            print(f"Training start model not found: {training_start_model}", flush=True)
            sys.exit(2)

        output_project = output_model.parent / "runs"
        run_name = output_model.stem

        print(f"ROOT: {ROOT}", flush=True)
        print(f"TRAINING_START_MODEL: {training_start_model}", flush=True)
        print(f"DATA_YAML: {data_yaml}", flush=True)
        print(f"OUTPUT_MODEL: {output_model}", flush=True)
        print(f"OUTPUT_PROJECT: {output_project}", flush=True)
        print(f"RUN_NAME: {run_name}", flush=True)
        print(f"EPOCHS: {DEFAULT_EPOCHS}", flush=True)
        print(f"PATIENCE: {DEFAULT_PATIENCE}", flush=True)
        print(f"IMGSZ: {DEFAULT_IMGSZ}", flush=True)
        print(f"BATCH: {DEFAULT_BATCH}", flush=True)
        print(f"REQUESTED_DEVICE: {device if device else 'auto'}", flush=True)
        print(f"RESOLVED_DEVICE: {resolved_device if resolved_device else 'auto'}", flush=True)

        model = YOLO(str(training_start_model))

        train_args = {
            "data": str(data_yaml),
            "imgsz": DEFAULT_IMGSZ,
            "epochs": DEFAULT_EPOCHS,
            "patience": DEFAULT_PATIENCE,
            "batch": DEFAULT_BATCH,
            "project": str(output_project),
            "name": run_name,
            "exist_ok": True,
            "pretrained": True,
            "plots": True,
            "val": True
        }

        if resolved_device:
            train_args["device"] = resolved_device

        model.train(**train_args)

        best_model = output_project / run_name / "weights" / "best.pt"

        if not best_model.exists():
            raise RuntimeError(f"Adapted model was not created: {best_model}")

        output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, output_model)

        print("Transfer learning completed.", flush=True)
        print(f"Best model from training: {best_model}", flush=True)
        print(f"Adapted model copied to: {output_model}", flush=True)

    except Exception:
        print("ERROR in retrain_model.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()