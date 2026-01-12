# Automated Quality Inspection System for PCB[ Task 1 & 2]

## Overview
This system uses a Faster R-CNN (ResNet-18 backbone) model to detect defects on PCB images.
It classifies defects into categories like `missing_hole`, `mouse_bite`, `short`, etc., and assigns a severity level (CRITICAL/MINOR).

## Prerequisites
- **Python 3.8+**
- **uv** (fast Python package installer and runner)

## Files
- `inspection_system.py`: Main script.
- `resnet18_frcnn_scratch.pth`: Trained model weights. [Download model and demo video](https://drive.google.com/drive/folders/1gVt0P94xHaQFHXgLMYR1Q5WEe8kqNDG-?usp=sharing)
- `demo.png`: Test image.

## Usage

Run the inspection script using `uv`:

```bash
uv run --with torch --with torchvision --with opencv-python --with matplotlib --with Pillow python inspection_system.py
```

## Output
- **Console**: Displays a text report of detected defects.
- **Image**: Saves `inspection_result.jpg` with bounding boxes and annotations.

---

# 3. Custom VLM Design for Industrial Quality Inspection [Task 3]

## Scenario
**Goal:** Offline PCB inspection where inspectors ask natural language questions about defects.
**Context:** 50,000 PCB images with BBoxes (no QA pairs). <2s inference latency required. Generic VLMs hallucinate.

## (A) Model Selection
**Recommendation:** **Qwen-VL-Chat (Int4 Quantized)** or **Custom NanoLLaVA (Phi-2 backbone)**.

*   **Primary Choice: Qwen-VL-Chat**
    *   **Reasoning:** It is natively trained for grounding (outputting bounding boxes). Unlike standard LLaVA which focuses on captions, Qwen-VL understands absolute coordinates out-of-the-box.
    *   **Resolution:** Supports 448x448 resolution natively (vs LLaVA's 336x336), enabling better detection of small PCB defects.
    *   **Licensing:** Apache 2.0 (Commercial friendly).

*   **Alternative: Custom TinyLLaVA (Phi-2 + SigLIP)**
    *   **Reasoning:** If the <2s constraint is hard on limited hardware (e.g., Jetson Nano), a 2.7B parameter model (Phi-2) is significantly faster than the 7B standard.

## (B) Design Strategy (PCB Specifics)
Standard VLMs fail on small defects (scratches, shorts).
1.  **Architecture Modification: Dynamic Tiling (The "Zoom" approach)**
    *   **Problem:** Resizing a 4K PCB image to 448px deletes the defect information.
    *   **Solution:** Use a dynamic high-resolution strategy (like LLaVA-Next or AnyRes). Split the image into a grid (e.g., 2x2 or 3x3).
    *   **Fusion:** Encode each tile separately using the Vision Encoder. Concatenate these "local features" with a downsampled "global view" feature map before feeding to the LLM. This provides both global context (location on board) and local high-res detail (defect shape).
2.  **Positional Encoding:**
    *   Inject explicit special tokens indicating the tile coordinates (e.g., `<tile_0_0>`, `<tile_0_1>`) to help the LLM reconstruct the geometry.

## (C) Optimization (<2s Inference)
1.  **4-bit Quantization (AWQ/GPTQ):**
    *   Compress the 7B model to ~4-5GB VRAM. This improves memory bandwidth utilization, directly speeding up token generation.
2.  **TensorRT-LLM / vLLM:**
    *   Do not use standard PyTorch eager mode. Compile the model with TensorRT-LLM for optimized kernel fusion and efficient attention mechanisms (FlashAttention-2).
3.  **Stop Token Optimization:**
    *   Since we need structured outputs, we can reduce latency by tuning the generation config to output concise JSON/Tuple formats and stop immediately, avoiding "chatty" explanations.

## (D) Hallucination Mitigation
1.  **Constrained Decoding (Grammar-Guided):**
    *   Use libraries like `guidance` or `outlines` to force the VLM output to strictly follow a schema (e.g., JSON).
    *   *Constraint:* The model *cannot* output text that isn't a valid defect class or coordinate pair. This physically prevents open-ended linguistic hallucinations.
2.  **Calibrated "Negative" Training:**
    *   Explicitly train the model to say "No defect found" or "Normal". Use the 50k images to generate "Negative Samples" (e.g., asking for a 'short' on a clean board). This reduces the False Positive rate.

## (E) Training Plan (Multi-Stage)
Since we lack QA pairs, we generate them synthetically from BBox data.

### 1. Data Fabrication
Generate a unified instruction tuning dataset (~150k samples):
*   **Type 1 (Localization):** Q: "Detect all mouse bites." A: `<box_200_300_250_350>`
*   **Type 2 (Existence - Positive):** Q: "Is there a missing hole?" A: "Yes, confidence 98% at <box>."
*   **Type 3 (Existence - Negative):** Q: "Is there a short?" A: "No defects detected."
*   **Type 4 (Counting):** Q: "Count the defects." A: "3 defects found."

### 2. Training Pipeline
*   **Stage 1: Projector Warmup (Optional):** If using a custom architecture (e.g., Phi-2 + SigLIP), freeze both and train only the projection layer (MLP) on the synthesized data.
*   **Stage 2: Supervised Fine-Tuning (SFT) with LoRA:**
    *   Freeze Vision Encoder.
    *   Fine-tune LLM using LoRA (Rank=64, Alpha=128).
    *   **Loss Function:** Standard Cross-Entropy, but apply a **Loss Mask** to zero out loss on the "User Question" tokens, focusing 100% on the "Answer" and "Coordinates".

## (F) Validation
1.  **Localization Precision:**
    *   **Metric:** mIoU (Mean Intersection over Union).
    *   **Threshold:** A detection is "Correct" if IoU > 0.5 with ground truth.
2.  **Hallucination Rate:**
    *   Feed the model 1,000 clean crop images and ask for defects.
    *   Calculate % of images where model predicts a bounding box (should be ~0%).
3.  **Inference Speed:**
    *   Measure end-to-end latency (Image Preprocessing + Encoder + Generation) on target hardware.
