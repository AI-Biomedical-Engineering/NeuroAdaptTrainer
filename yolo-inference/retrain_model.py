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
import csv

ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "models" / "best.pt"

DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 35
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 1
DEFAULT_SAVE_PERIOD = 5
CLEAN_EPOCH_CHECKPOINTS = True
PRIMARY_METRIC = "metrics/mAP50(M)"
FALLBACK_METRICS = [
    "metrics/mAP50(M)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95(B)",
]

def as_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def find_best_checkpoint_by_metric(run_dir):
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"

    if not results_csv.exists() or not weights_dir.exists():
        return None, None, None

    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {key.strip(): value for key, value in row.items()}
            for row in reader
        ]

    if not rows:
        return None, None, None

    fieldnames = rows[0].keys()

    selected_metric = None
    for metric in FALLBACK_METRICS:
        if metric in fieldnames:
            selected_metric = metric
            break

    if selected_metric is None:
        return None, None, None

    best_checkpoint = None
    best_value = None
    best_epoch = None

    for row in rows:
        value = as_float_or_none(row.get(selected_metric))

        if value is None:
            continue

        epoch = int(float(row["epoch"]))
        checkpoint = weights_dir / f"epoch{epoch}.pt"

        # Con save_period=10 solo existen algunos epoch*.pt.
        # Por eso solo se consideran épocas cuyo checkpoint fue guardado.
        if not checkpoint.exists():
            continue

        if best_value is None or value > best_value:
            best_value = value
            best_checkpoint = checkpoint
            best_epoch = epoch

    if best_checkpoint is None:
        return None, selected_metric, None

    print(
        f"Best saved checkpoint epoch: {best_epoch}",
        flush=True
    )

    return best_checkpoint, selected_metric, best_value

def clean_epoch_checkpoints(run_dir):
    weights_dir = run_dir / "weights"

    if not weights_dir.exists():
        return

    deleted = 0

    for checkpoint in weights_dir.glob("epoch*.pt"):
        try:
            checkpoint.unlink()
            deleted += 1
        except OSError as e:
            print(
                f"Could not delete checkpoint {checkpoint}: {e}",
                flush=True
            )

    print(
        f"Deleted {deleted} intermediate epoch checkpoints.",
        flush=True
    )

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
        print(f"SAVE_PERIOD: {DEFAULT_SAVE_PERIOD}", flush=True)
        print(f"CLEAN_EPOCH_CHECKPOINTS: {CLEAN_EPOCH_CHECKPOINTS}", flush=True)
        print(f"REQUESTED_DEVICE: {device if device else 'auto'}", flush=True)
        print(f"RESOLVED_DEVICE: {resolved_device if resolved_device else 'auto'}", flush=True)
        print(f"PRIMARY_METRIC: {PRIMARY_METRIC}", flush=True)

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
            "val": True,
            "save_period": DEFAULT_SAVE_PERIOD,
        }

        if resolved_device:
            train_args["device"] = resolved_device

        model.train(**train_args)

        run_dir = output_project / run_name
        ultralytics_best_model = run_dir / "weights" / "best.pt"

        selected_model, selected_metric, selected_value = find_best_checkpoint_by_metric(run_dir)

        if selected_model is None:
            selected_model = ultralytics_best_model
            print(
                f"Could not select checkpoint by {PRIMARY_METRIC}. Falling back to Ultralytics best.pt.",
                flush=True
            )
        else:
            print(
                f"Selected checkpoint by {selected_metric}: {selected_model} "
                f"(value={selected_value:.6f})",
                flush=True
            )

        if not selected_model.exists():
            raise RuntimeError(f"Adapted model was not created: {selected_model}")

        output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_model, output_model)

        if CLEAN_EPOCH_CHECKPOINTS:
            clean_epoch_checkpoints(run_dir)

        print("Transfer learning completed.", flush=True)
        print(f"Ultralytics best model: {ultralytics_best_model}", flush=True)
        print(f"Selected model copied from: {selected_model}", flush=True)
        print(f"Adapted model copied to: {output_model}", flush=True)
        print(f"PRIMARY_METRIC: {PRIMARY_METRIC}", flush=True)

    except Exception:
        print("ERROR in retrain_model.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()