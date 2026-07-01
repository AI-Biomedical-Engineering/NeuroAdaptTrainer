#!/bin/bash

set -e

echo "Installing Neuron Segmentation Assistant dependencies..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INFERENCE_DIR="$PROJECT_ROOT/yolo-inference"
VENV_DIR="$PROJECT_ROOT/venv"
CONFIG_DIR="$HOME/.neuron-segmentation-assistant"
CONFIG_FILE="$CONFIG_DIR/config.properties"
COMPARE_SCRIPT="$INFERENCE_DIR/compare_models.py"

echo "Project root: $PROJECT_ROOT"
echo "Inference directory: $INFERENCE_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not installed."
    echo "Please install Python 3 first."
    exit 1
fi

echo "Creating Python virtual environment..."

python3 -m venv "$VENV_DIR"

echo "Activating virtual environment..."

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."

python -m pip install --upgrade pip

echo "Installing Python dependencies..."

pip install -r "$INFERENCE_DIR/requirements.txt"

echo "Creating configuration file..."

mkdir -p "$CONFIG_DIR"

cat > "$CONFIG_FILE" <<EOF
python=$VENV_DIR/bin/python
script=$INFERENCE_DIR/infer_one.py
retrain_script=$INFERENCE_DIR/retrain_model.py
compare_script=$INFERENCE_DIR/compare_models.py
EOF

echo ""
echo "Installation completed successfully."
echo "Configuration saved to:"
echo "$CONFIG_FILE"
echo ""
echo "You can now open Fiji and run:"
echo "Plugins > Neuron Segmentation > Single Image Segmentation"
echo "or:"
echo "Plugins > Neuron Segmentation > Transfer Learning Assistant"
echo "or:"
echo "Plugins > Neuron Segmentation > Model Comparison / External Validation"