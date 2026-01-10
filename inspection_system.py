import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

# --- Architecture Setup ---
def get_resnet18_faster_rcnn(num_classes):
    """
    Re-defines the model architecture exactly as trained.
    backbone: ResNet-18 (weights=None)
    out_channels: 512
    """
    # Load a pre-trained model for the architecture, but we'll replace the backbone?
    # Actually, user said "ResNet-18 backbone (trained from scratch) ... weights=None"
    # The standard torchvision FasterRCNN uses ResNet-50.
    # We need to construct it manually to match "trained from scratch" with Resnet18.
    
    # 1. Load ResNet18 backbone
    backbone = torchvision.models.resnet18(weights=None)
    
    # Remove the last two layers (avgpool and fc) to use as a backbone for Faster R-CNN
    # ResNet18 output channels at layer4 is 512.
    modules = list(backbone.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    
    # FasterRCNN needs the backbone to have an 'out_channels' attribute
    backbone.out_channels = 512
    
    # 2. Create the Anchor Generator
    # We need to match the anchor generator used during training.
    # Default is usually sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    # Since specific training params weren't provided other than "scratch", 
    # we'll try the standard wrapper which handles RPN defaults.
    # However, strictly speaking, `fasterrcnn_resnet50_fpn` uses FPN. 
    # A simple "ResNet18 backbone" might imply no FPN or a custom build.
    # 'torchvision.models.detection.FasterRCNN' is the generic class.
    
    # Let's assume the user used a common pattern for "resnet18 frcnn":
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

# --- Configuration ---
CLASSES = ['background', 'missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
MODEL_PATH = 'resnet18_frcnn_scratch.pth'
IMAGE_PATH = 'input.jpeg'

# --- Inspection Logic ---
def analyze_product(image_path, model, device):
    # Load image
    img = Image.open(image_path).convert("RGB")
    # Convert to tensor
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    # Filter results
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    results = []
    
    for i in range(len(scores)):
        score = scores[i]
        if score < 0.3:
            continue
            
        box = boxes[i]
        label_id = labels[i]
        label_name = CLASSES[label_id] if label_id < len(CLASSES) else f"Unknown({label_id})"
        
        # Calculate Defect Center
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        area = (x2 - x1) * (y2 - y1)
        
        # Assign Severity
        # HEURISTIC: CRITICAL if confidence > 0.8 OR area > 1500 pixels
        if score > 0.8 or area > 1500:
            severity = "CRITICAL"
        else:
            severity = "MINOR"
            
        results.append({
            'box': box,
            'score': score,
            'label': label_name,
            'center': (center_x, center_y),
            'severity': severity
        })
        
    return img, results

# --- Visualization ---
def visualize_results(image, results):
    # Convert PIL to OpenCV format (RGB -> BGR) for drawing, 
    # but strictly we can draw on the array and then plot with matplotlib (keeps RGB)
    img_np = np.array(image)
    
    # Matplotlib setup
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)
    
    print("-" * 60)
    print(f"{'Defect Log':^60}")
    print("-" * 60)
    
    for i, res in enumerate(results):
        box = res['box']
        score = res['score']
        label = res['label']
        cx, cy = res['center']
        severity = res['severity']
        
        # Format text report
        print(f"Defect #{i+1}: Type={label}, Center=({cx}, {cy}), Severity={severity}, Score={score:.2f}")
        
        # Draw Box
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Color based on severity
        edge_color = 'r' if severity == "CRITICAL" else 'y'
        
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)
        
        # Draw Center Point
        ax.plot(cx, cy, 'bo', markersize=5)
        
        # Labels
        # Label text: "Mouse Bite | CRITICAL"
        label_text = f"{label} | {severity}"
        # Overlay coordinates near center
        coord_text = f"({cx},{cy})"
        
        ax.text(x1, y1 - 5, label_text, color=edge_color, fontsize=8, backgroundcolor='white')
        ax.text(cx + 5, cy, coord_text, color='blue', fontsize=7)

    print("-" * 60)
    
    plt.axis('off')
    plt.title("Automated Quality Inspection Result")
    plt.tight_layout()
    
    # Save result for headless verification
    plt.savefig('inspection_result.jpg')
    print("Result saved to 'inspection_result.jpg'")
    # prompt user to close if running interactively
    # plt.show() # Commented out to prevent blocking in some environments, but useful if local.

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        exit(1)
        
    num_classes = len(CLASSES)
    model = get_resnet18_faster_rcnn(num_classes)
    
    try:
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Tip: If keys mismatch, ensure the architecture definition matches exactly how it was saved.")
        exit(1)
        
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Analyze
    if not os.path.exists(IMAGE_PATH):
         print(f"Error: Image file '{IMAGE_PATH}' not found.")
         exit(1)

    print(f"Analyzing {IMAGE_PATH}...")
    image, inspection_results = analyze_product(IMAGE_PATH, model, device)
    
    # 3. Present Results
    visualize_results(image, inspection_results)
