"""
retrain_model.py

Retrains/adapts the YOLO neuron segmentation model using corrected
annotations exported from the Fiji/ImageJ plugin.

Usage:
    python retrain_model.py <data_yaml_path>
"""

from pathlib import Path
import sys
import traceback
import shutil
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

BASE_MODEL = ROOT / "models" / "best.pt"
ADAPTED_MODEL = ROOT / "models" / "best_adapted.pt"

TRAINING_START_MODEL = ADAPTED_MODEL if ADAPTED_MODEL.exists() else BASE_MODEL

OUTPUT_PROJECT = ROOT / "runs" / "transfer_learning"


def main():
    try:
        if len(sys.argv) < 2:
            print("Usage: python retrain_model.py <data_yaml_path>", flush=True)
            sys.exit(2)

        data_yaml = Path(sys.argv[1]).expanduser().resolve()

        if not data_yaml.exists():
            print(f"data.yaml not found: {data_yaml}", flush=True)
            sys.exit(2)

        if not TRAINING_START_MODEL.exists():
            print(f"Training start model not found: {TRAINING_START_MODEL}", flush=True)
            sys.exit(2)

        print(f"ROOT: {ROOT}", flush=True)
        print(f"TRAINING_START_MODEL: {TRAINING_START_MODEL}", flush=True)
        print(f"DATA_YAML: {data_yaml}", flush=True)
        print(f"OUTPUT_PROJECT: {OUTPUT_PROJECT}", flush=True)

        model = YOLO(str(TRAINING_START_MODEL))

        results = model.train(
            data=str(data_yaml),
            imgsz=640,
            epochs=5,
            batch=2,
            project=str(OUTPUT_PROJECT),
            name="adapted_model",
            exist_ok=True,
            pretrained=True
        )

        best_model = OUTPUT_PROJECT / "adapted_model" / "weights" / "best.pt"

        if not best_model.exists():
            raise RuntimeError(f"Adapted model was not created: {best_model}")

        ADAPTED_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, ADAPTED_MODEL)

        print("Transfer learning completed.", flush=True)
        print(f"Best model from training: {best_model}", flush=True)
        print(f"Adapted model copied to: {ADAPTED_MODEL}", flush=True)

    except Exception:
        print("ERROR in retrain_model.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()