from pathlib import Path
import sys
import csv
import cv2
import numpy as np


def draw_yolo_txt(image, label_path):
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for idx, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) < 7:
            print(f"Skipping invalid YOLO line {idx}: {line}")
            continue

        coords = list(map(float, parts[1:]))

        points = []
        for i in range(0, len(coords) - 1, 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            points.append([x, y])

        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.int32)

        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

        cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(idx),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )


def draw_csv_detection(image, csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            cx = float(row["cx"])
            cy = float(row["cy"])
            radius = float(row["radius"])

            cv2.circle(
                image,
                (int(round(cx)), int(round(cy))),
                int(round(radius)),
                (255, 0, 0),
                2
            )

            cv2.circle(
                image,
                (int(round(cx)), int(round(cy))),
                3,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                str(idx),
                (int(round(cx)) + 4, int(round(cy)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("python verify_annotation_overlay.py <image_path> <annotation_txt_or_csv> <output_image>")
        sys.exit(2)

    image_path = Path(sys.argv[1]).expanduser().resolve()
    annotation_path = Path(sys.argv[2]).expanduser().resolve()
    output_path = Path(sys.argv[3]).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not open image with OpenCV: {image_path}")

    if annotation_path.suffix.lower() == ".txt":
        draw_yolo_txt(image, annotation_path)
    elif annotation_path.suffix.lower() == ".csv":
        draw_csv_detection(image, annotation_path)
    else:
        raise RuntimeError("Annotation must be .txt or .csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    print(f"Verification image saved to: {output_path}")


if __name__ == "__main__":
    main()