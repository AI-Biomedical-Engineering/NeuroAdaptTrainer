"""
retrain_model.py

Retrains/adapts the YOLO neuron segmentation model using annotations
generated from the Fiji/ImageJ transfer learning module.

Usage:
    python retrain_model.py <data_yaml_path> <output_model_path> [start_model_path]
"""

from pathlib import Path
import sys
import traceback
import shutil
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "best.pt"


def main():
    try:
        if len(sys.argv) < 3:
            print(
                "Usage: python retrain_model.py <data_yaml_path> <output_model_path> [start_model_path]",
                flush=True
            )
            sys.exit(2)

        data_yaml = Path(sys.argv[1]).expanduser().resolve()
        output_model = Path(sys.argv[2]).expanduser().resolve()

        if len(sys.argv) >= 4:
            training_start_model = Path(sys.argv[3]).expanduser().resolve()
        else:
            training_start_model = BASE_MODEL

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

        model = YOLO(str(training_start_model))

        model.train(
            data=str(data_yaml),
            imgsz=640,
            epochs=5,
            batch=2,
            project=str(output_project),
            name=run_name,
            exist_ok=True,
            pretrained=True
        )

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