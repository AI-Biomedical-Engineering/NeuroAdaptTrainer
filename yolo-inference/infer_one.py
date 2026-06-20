"""
infer_one.py

This module performs single-image neuron segmentation using a trained
Ultralytics YOLO model. It receives an input image path (and optionally
an output path), runs segmentation inference, applies post-processing
(area filtering and NMS), and generates a visualization with detected
neurons marked as circles.

Intended usage:
    python infer_one.py <input_image_path> [output_image_path]

If an output path is provided (e.g., from Fiji), the processed image
is written there. Additionally, a stable and a timestamped output are
stored inside the project outputs directory.
"""
from ultralytics import YOLO
from pathlib import Path
import sys
import cv2
import numpy as np
import traceback

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# ROOT is the folder where this script lives, independent of the current working directory.
ROOT = Path(__file__).resolve().parent

BASE_MODEL_PATH = ROOT / "models" / "best.pt"


def resolve_model_path():
    """
    Resolve the model path used for inference.

    Usage:
        python infer_one.py <image_path> [output_path] [model_path]

    If model_path is provided, it is used.
    Otherwise, the base model is used.
    """
    if len(sys.argv) >= 4:
        return Path(sys.argv[3]).expanduser().resolve()

    return BASE_MODEL_PATH

# -----------------------------------------------------------------------------
# Inference / Visualization configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 640
CONF_THRES = 0.25

# Drawing parameters (OpenCV uses BGR)
BLUE = (255, 0, 0)
THICKNESS = 1
LINE_TYPE = cv2.LINE_AA

# Optional post-processing
APPLY_CONTRAST = False
CONTRAST_ALPHA = 1.08
CONTRAST_BETA = 4

# Filtering and de-duplication
MIN_AREA = 25
NMS_IOU = 0.35

# Polygon export
ELLIPSE_POLYGON_POINTS = 32
MASK_POLYGON_EPSILON_RATIO = 0.002

# Visualization style
DRAW_STYLE = "circle"
DRAW_CENTER = True
RADIUS_MIN = 3
RADIUS_MAX = 14


def safe_imwrite(path: Path, img: np.ndarray) -> None:
    """
    Write an image to disk and validate that it was created correctly.
    Raises an exception if writing fails or the file is empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok or (not path.exists()) or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write: {path}")

def save_detections_csv(path: Path, items):
    """
    Save detected neurons as CSV.

    Columns:
        cx,cy,radius,ellipse_polygon,mask_polygon

    The first three columns are kept for compatibility with the Java plugin.
    ellipse_polygon stores a 32-point circular approximation.
    mask_polygon stores the simplified contour extracted from the original YOLO mask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("cx,cy,radius,ellipse_polygon,mask_polygon\n")

        for (box, (cx, cy), area, mask) in items:
            rad = int(round(np.sqrt(max(area, 1) / np.pi)))
            rad = max(RADIUS_MIN, min(RADIUS_MAX, rad))

            ellipse_polygon = ellipse_polygon_from_detection(
                cx=cx,
                cy=cy,
                radius=rad,
                points=ELLIPSE_POLYGON_POINTS
            )

            mask_polygon = mask_polygon_from_binary_mask(mask)

            ellipse_polygon_str = serialize_polygon(ellipse_polygon)
            mask_polygon_str = serialize_polygon(mask_polygon)

            f.write(
                f"{cx},{cy},{rad},"
                f"\"{ellipse_polygon_str}\","
                f"\"{mask_polygon_str}\"\n"
            )

def filter_masks_by_area(mask_tensor: np.ndarray, min_area: int) -> np.ndarray:
    """
    Keep only masks whose binarized area is >= min_area.
    mask_tensor shape: (N, H, W).
    """
    if mask_tensor.size == 0:
        return mask_tensor
    kept = []
    for i in range(mask_tensor.shape[0]):
        m_bin = (mask_tensor[i] > 0.5).astype(np.uint8)
        if int(m_bin.sum()) >= min_area:
            kept.append(mask_tensor[i])
    if not kept:
        return np.zeros((0, mask_tensor.shape[1], mask_tensor.shape[2]), dtype=mask_tensor.dtype)
    return np.stack(kept, axis=0)

def masks_to_boxes_centroids_and_masks(mask_tensor: np.ndarray, target_width: int, target_height: int):
    """
    Convert masks to (bounding box, centroid, area, binary_mask) tuples for each instance.

    The masks are resized to the original image size before extracting coordinates.
    This ensures that exported polygons are aligned with the image saved by the Java plugin.
    """
    items = []

    for i in range(mask_tensor.shape[0]):
        m = (mask_tensor[i] > 0.5).astype(np.uint8)

        if m.shape[1] != target_width or m.shape[0] != target_height:
            m = cv2.resize(
                m,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )

        ys, xs = np.where(m > 0)

        if xs.size == 0 or ys.size == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        area = int(xs.size)
        cx = float(xs.mean())
        cy = float(ys.mean())

        items.append(((x1, y1, x2, y2), (cx, cy), area, m))

    return items

def iou_xyxy(a, b) -> float:
    """
    Compute Intersection over Union (IoU) between two xyxy bounding boxes.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_on_items(items, iou_thres: float):
    """
    Simple non-maximum suppression using area as a proxy score.
    Larger instances are kept when overlap exceeds iou_thres.
    """
    if not items:
        return items
    items = sorted(items, key=lambda t: t[2], reverse=True)  # sort by area desc
    kept = []
    for cand in items:
        cand_box = cand[0]
        if all(iou_xyxy(cand_box, k[0]) < iou_thres for k in kept):
            kept.append(cand)
    return kept


def clamp_xyxy(box, w, h):
    """
    Clamp a bounding box to image boundaries.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def ellipse_polygon_from_detection(cx: float, cy: float, radius: int, points: int = ELLIPSE_POLYGON_POINTS):
    """
    Generate a polygonal approximation of a circle centered at (cx, cy).
    The result is a list of absolute image coordinates:
        [x1, y1, x2, y2, ..., xn, yn]
    """
    coords = []

    for i in range(points):
        angle = 2.0 * np.pi * i / points

        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)

        coords.extend([float(x), float(y)])

    return coords


def mask_polygon_from_binary_mask(mask: np.ndarray, epsilon_ratio: float = MASK_POLYGON_EPSILON_RATIO):
    """
    Extract the largest contour from a binary mask and approximate it as a polygon.
    The result is a list of absolute image coordinates:
        [x1, y1, x2, y2, ..., xn, yn]
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter <= 0:
        return []

    epsilon = epsilon_ratio * perimeter

    approximated = cv2.approxPolyDP(
        largest_contour,
        epsilon,
        True
    )

    coords = []

    for point in approximated:
        x, y = point[0]
        coords.extend([float(x), float(y)])

    return coords


def serialize_polygon(coords):
    """
    Serialize a polygon coordinate list into a compact string.
    Example:
        [10.2, 15.4, 20.0, 30.0] -> "10.20 15.40 20.00 30.00"
    """
    return " ".join(f"{value:.2f}" for value in coords)

def main():
    """
    Entry point:
    - argv[1]: input image path
    - argv[2] (optional): output image path (used by Fiji)
    The script writes:
    - ROOT/outputs/latest_neurons_clean.png (stable output)
    - argv[2] if provided (temporary output for Fiji)
    - ROOT/outputs/neurons_<timestamp>.png (versioned output)
    """
    try:
        if len(sys.argv) < 2:
            print("Usage: python infer_one.py <image_path> [output_path]", file=sys.stderr, flush=True)
            sys.exit(2)

        image_path = Path(sys.argv[1]).expanduser().resolve()
        if not image_path.exists():
            print(f"Input image not found: {image_path}", file=sys.stderr, flush=True)
            sys.exit(2)
            
        model_path = resolve_model_path()

        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr, flush=True)
            sys.exit(2)

        # Optional output path provided by Fiji (typically a /tmp file)
        out_override = None
        if len(sys.argv) >= 3:
            out_override = Path(sys.argv[2]).expanduser().resolve()

        # Select a writable directory for Ultralytics internal artifacts.
        # This avoids the default "runs/" directory which may fail under Fiji.
        stable_dir = (out_override.parent if out_override else (ROOT / "outputs"))
        stable_dir.mkdir(parents=True, exist_ok=True)

        print(f"ROOT: {ROOT}", flush=True)
        print(f"MODEL: {model_path}", flush=True)
        print(f"INPUT: {image_path}", flush=True)
        print(f"ULTRA_DIR: {stable_dir}", flush=True)

        model = YOLO(str(model_path))

        # Run YOLO inference (do not save Ultralytics default images; we draw our own output)
        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            save=False,
            verbose=False,
            project=str(stable_dir),
            name="ultralytics_tmp",
            exist_ok=True,
        )

        r = results[0]

        # Use the original image without Ultralytics overlays as the base canvas.
        base_rgb = r.plot(labels=False, conf=False, boxes=False, masks=False)
        base_rgb = np.asarray(base_rgb).astype(np.uint8)
        image_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

        if APPLY_CONTRAST:
            image_bgr = cv2.convertScaleAbs(image_bgr, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)

        # If segmentation masks exist, post-process them and export detections to CSV.
        if r.masks is not None and len(r.masks.data) > 0:
            mask_np = r.masks.data.detach().cpu().numpy()
            mask_np = filter_masks_by_area(mask_np, MIN_AREA)

            image_height, image_width = image_bgr.shape[:2]
            items = masks_to_boxes_centroids_and_masks(mask_np, image_width, image_height)
            items = nms_on_items(items, NMS_IOU)

            print(f"Detected neurons (filtered + NMS): {len(items)}", flush=True)

            csv_path = stable_dir / "imagej_output.csv"
            save_detections_csv(csv_path, items)
            print(f"CSV detections saved to: {csv_path}", flush=True)

        else:
            print("Detected neurons (filtered + NMS): 0", flush=True)

            items = []
            csv_path = stable_dir / "imagej_output.csv"
            save_detections_csv(csv_path, items)
            print(f"CSV detections saved to: {csv_path}", flush=True)
        # Write Fiji temporary output. This is the file opened by the Java plugin.
        if out_override:
            safe_imwrite(out_override, image_bgr)
            print(f"Output saved to: {out_override}", flush=True)
        else:
            # Manual execution fallback: only save to outputs if no output path was provided.
            stable_out = ROOT / "outputs" / "latest_neurons_clean.png"
            safe_imwrite(stable_out, image_bgr)
            print(f"Output saved to: {stable_out}", flush=True)
                
    except Exception:
        # Print the full traceback so Fiji can display the real error.
        print("ERROR in infer_one.py:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()