#!/bin/bash

# Default configuration
IMAGE_PATH="./path_to_your_image.jpg"  # Path to your input image
TEXT="A description of the image you want to analyze"  # Text prompt for CLIP model
ARCH="ViT-B-16"  # CLIP model architecture
DEVICE="cuda"  # Set to 'cuda' or 'cpu' depending on your hardware
OUTPUT_DIR="./output_heatmaps"  # Directory to save generated heatmaps

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Activate your Python environment if needed
# source /path/to/your/conda/env/bin/activate

# Run the Python script
python script_name.py \
  --image "$IMAGE_PATH" \
  --text "$TEXT" \
  --arch "$ARCH" \
  --device "$DEVICE" \
  --output "$OUTPUT_DIR"

echo "Word-level heatmaps have been generated and saved in the $OUTPUT_DIR directory."
