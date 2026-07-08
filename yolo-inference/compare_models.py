"""
compare_models.py

Compares a base YOLO model against a transfer learning model using an
external validation dataset annotated in YOLO format.

Usage:
    python compare_models.py <base_model_path> <adapted_model_path> <data_yaml_path> <output_dir> [device]

Device values:
    cpu
    cuda
    mps
    auto_acceleration
"""

from pathlib import Path
import sys
import csv
import traceback
from ultralytics import YOLO
import torch


DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 1
DEFAULT_WORKERS = 0
PRIMARY_METRIC = "metrics/mAP50(M)"
FALLBACK_METRICS = [
    "metrics/mAP50(M)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95(B)",
]


PREFERRED_METRICS = [
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
]


def cuda_is_usable():
    if not torch.cuda.is_available():
        return False

    try:
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception as e:
        print(
            f"CUDA is available but could not be used: {e}. Falling back to CPU.",
            flush=True
        )
        return False


def resolve_device(device):
    if device is None:
        return None

    if device == "auto_acceleration":
        if cuda_is_usable():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        print(
            "Hardware acceleration was requested, but CUDA/MPS is not available. Falling back to CPU.",
            flush=True
        )
        return "cpu"

    if device == "cuda" and not cuda_is_usable():
        print(
            "CUDA was requested but is not usable. Falling back to CPU.",
            flush=True
        )
        return "cpu"

    return device


def as_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_model(model_path, data_yaml, output_dir, run_name, device=None):
    model = YOLO(str(model_path))

    val_args = {
        "data": str(data_yaml),
        "imgsz": DEFAULT_IMGSZ,
        "batch": DEFAULT_BATCH,
        "workers": DEFAULT_WORKERS,
        "plots": True,
        "save_json": False,
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    if device:
        val_args["device"] = device

    results = model.val(**val_args)

    metrics = {}

    if hasattr(results, "results_dict"):
        metrics.update(results.results_dict)

    return metrics


def format_metric_value(value):
    numeric = as_float_or_none(value)

    if numeric is None:
        return value

    return f"{numeric:.6f}"


def write_metrics_csv(output_file, base_metrics, adapted_metrics):
    all_keys = sorted(set(base_metrics.keys()) | set(adapted_metrics.keys()))

    preferred = [key for key in PREFERRED_METRICS if key in all_keys]
    remaining = [key for key in all_keys if key not in preferred]
    ordered_keys = preferred + remaining

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "base_model", "adapted_model", "difference"])

        for key in ordered_keys:
            base_value = base_metrics.get(key, "")
            adapted_value = adapted_metrics.get(key, "")

            base_number = as_float_or_none(base_value)
            adapted_number = as_float_or_none(adapted_value)

            difference = ""

            if base_number is not None and adapted_number is not None:
                difference = adapted_number - base_number

            writer.writerow([
                key,
                format_metric_value(base_value),
                format_metric_value(adapted_value),
                format_metric_value(difference) if difference != "" else ""
            ])


def pick_metric(metrics, names):
    for name in names:
        if name in metrics:
            return name, metrics[name]

    return None, None


def write_summary(output_file, base_model, adapted_model, data_yaml, base_metrics, adapted_metrics, requested_device, resolved_device):
    rows = []

    for metric_name in PREFERRED_METRICS:
        if metric_name in base_metrics or metric_name in adapted_metrics:
            base_value = base_metrics.get(metric_name, "")
            adapted_value = adapted_metrics.get(metric_name, "")

            base_number = as_float_or_none(base_value)
            adapted_number = as_float_or_none(adapted_value)

            difference = ""

            if base_number is not None and adapted_number is not None:
                difference = adapted_number - base_number

            rows.append((metric_name, base_value, adapted_value, difference))

    best_map_name, adapted_map = pick_metric(adapted_metrics, FALLBACK_METRICS)

    base_map = base_metrics.get(best_map_name, None) if best_map_name else None

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Model comparison summary\n")
        f.write("========================\n\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Adapted model: {adapted_model}\n")
        f.write(f"Validation data: {data_yaml}\n")
        f.write(f"Requested device: {requested_device if requested_device else 'auto'}\n")
        f.write(f"Resolved device: {resolved_device if resolved_device else 'auto'}\n")
        f.write(f"Image size: {DEFAULT_IMGSZ}\n")
        f.write(f"Batch size: {DEFAULT_BATCH}\n")
        f.write(f"Workers: {DEFAULT_WORKERS}\n\n")
        f.write(f"Preferred primary metric: {PRIMARY_METRIC}\n\n")

        if best_map_name is not None:
            f.write(f"Main metric: {best_map_name}\n")
            f.write(f"Base model: {format_metric_value(base_map)}\n")
            f.write(f"Adapted model: {format_metric_value(adapted_map)}\n")

            base_number = as_float_or_none(base_map)
            adapted_number = as_float_or_none(adapted_map)

            if base_number is not None and adapted_number is not None:
                f.write(f"Difference: {adapted_number - base_number:.6f}\n")

        f.write("\nSelected metrics:\n")

        for metric_name, base_value, adapted_value, difference in rows:
            f.write(
                f"- {metric_name}: "
                f"base={format_metric_value(base_value)}, "
                f"adapted={format_metric_value(adapted_value)}"
            )

            if difference != "":
                f.write(f", difference={format_metric_value(difference)}")

            f.write("\n")


def print_metrics(title, metrics):
    print(f"\n{title}", flush=True)

    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]}", flush=True)


def main():
    try:
        if len(sys.argv) < 5:
            print(
                "Usage: python compare_models.py <base_model_path> <adapted_model_path> <data_yaml_path> <output_dir> [device]",
                flush=True
            )
            sys.exit(2)

        base_model = Path(sys.argv[1]).expanduser().resolve()
        adapted_model = Path(sys.argv[2]).expanduser().resolve()
        data_yaml = Path(sys.argv[3]).expanduser().resolve()
        output_dir = Path(sys.argv[4]).expanduser().resolve()

        if len(sys.argv) >= 6:
            requested_device = sys.argv[5]
        else:
            requested_device = None

        resolved_device = resolve_device(requested_device)

        if not base_model.exists():
            raise RuntimeError(f"Base model not found: {base_model}")

        if not adapted_model.exists():
            raise RuntimeError(f"Adapted model not found: {adapted_model}")

        if not data_yaml.exists():
            raise RuntimeError(f"Validation data.yaml not found: {data_yaml}")

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Comparing models...", flush=True)
        print(f"BASE_MODEL: {base_model}", flush=True)
        print(f"ADAPTED_MODEL: {adapted_model}", flush=True)
        print(f"DATA_YAML: {data_yaml}", flush=True)
        print(f"OUTPUT_DIR: {output_dir}", flush=True)
        print(f"REQUESTED_DEVICE: {requested_device if requested_device else 'auto'}", flush=True)
        print(f"RESOLVED_DEVICE: {resolved_device if resolved_device else 'auto'}", flush=True)
        print(f"IMGSZ: {DEFAULT_IMGSZ}", flush=True)
        print(f"BATCH: {DEFAULT_BATCH}", flush=True)
        print(f"WORKERS: {DEFAULT_WORKERS}", flush=True)
        print(f"PRIMARY_METRIC: {PRIMARY_METRIC}", flush=True)

        print("\nValidating base model...", flush=True)
        base_metrics = validate_model(
            base_model,
            data_yaml,
            output_dir,
            "base_model_validation",
            resolved_device
        )

        print("\nValidating adapted model...", flush=True)
        adapted_metrics = validate_model(
            adapted_model,
            data_yaml,
            output_dir,
            "adapted_model_validation",
            resolved_device
        )

        output_csv = output_dir / "model_comparison_metrics.csv"
        output_summary = output_dir / "model_comparison_summary.txt"

        write_metrics_csv(output_csv, base_metrics, adapted_metrics)
        write_summary(
            output_summary,
            base_model,
            adapted_model,
            data_yaml,
            base_metrics,
            adapted_metrics,
            requested_device,
            resolved_device
        )

        print("\nComparison completed.", flush=True)
        print(f"Metrics saved to: {output_csv}", flush=True)
        print(f"Summary saved to: {output_summary}", flush=True)

        print_metrics("Base model metrics:", base_metrics)
        print_metrics("Adapted model metrics:", adapted_metrics)

    except Exception:
        print("ERROR in compare_models.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
