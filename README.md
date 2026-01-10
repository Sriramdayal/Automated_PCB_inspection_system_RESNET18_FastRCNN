# Automated Quality Inspection System for PCB

## Overview
This system uses a Faster R-CNN (ResNet-18 backbone) model to detect defects on PCB images.
It classifies defects into categories like `missing_hole`, `mouse_bite`, `short`, etc., and assigns a severity level (CRITICAL/MINOR).

## Prerequisites
- **Python 3.8+**
- **uv** (fast Python package installer and runner)

## Files
- `inspection_system.py`: Main script.
- `resnet18_frcnn_scratch.pth`: Trained model weights. [Download model](https://drive.google.com/drive/folders/1gVt0P94xHaQFHXgLMYR1Q5WEe8kqNDG-?usp=sharing)
- `download.png`: Test image.

## Usage

Run the inspection script using `uv`:

```bash
uv run --with torch --with torchvision --with opencv-python --with matplotlib --with Pillow python inspection_system.py
```

## Output
- **Console**: Displays a text report of detected defects.
- **Image**: Saves `inspection_result.jpg` with bounding boxes and annotations.
