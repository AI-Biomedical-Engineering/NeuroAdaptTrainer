# YOLO model

This folder contains the final trained YOLO segmentation model used by the Fiji/ImageJ plugin.

The file `best.pt` corresponds to the selected training run:

`200_AllExperts_rotated_2025_5_15`

The original model was located at:

`runs/segment/200_AllExperts_rotated_2025_5_15/weights/best.pt`

## Selected model metrics

The model was selected because it achieved the best segmentation performance among the trained candidates, especially for mask-based metrics:

- `metrics/mAP50-95(M)`: 0.4075
- `metrics/mAP50(M)`: 0.9228
- `metrics/recall(M)`: 0.8970
- `metrics/precision(M)`: 0.8707

## Training curves

![Training results](../../training-results/model-comparison/plots/200_AllExperts_rotated_2025_5_15.png)