import torch
import torchvision
import time
import argparse
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from inspection_system import get_resnet18_faster_rcnn, CLASSES

MODEL_PATH = 'resnet18_frcnn_scratch.pth'
IMAGE_PATH = 'input.jpeg'

# --- Dataset for mAP ---
class PCBDataset(Dataset):
    def __init__(self, root_dir, class_list):
        self.root_dir = root_dir
        self.class_list = class_list
        self.class_to_idx = {cls: i for i, cls in enumerate(class_list)}
        
        # Expect images and annotations folders
        self.image_dir = os.path.join(root_dir, 'images')
        self.annot_dir = os.path.join(root_dir, 'annotations')
        
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.annot_dir):
             raise ValueError(f"Data directory must contain 'images' and 'annotations' subdirectories. Found: {os.listdir(root_dir) if os.path.isdir(root_dir) else 'Path not found'}")

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.[jJ][pP][gG]')) + 
                                  glob.glob(os.path.join(self.image_dir, '*.[pP][nN][gG]')))
                                  
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        file_name = os.path.basename(img_path)
        xml_name = os.path.splitext(file_name)[0] + '.xml'
        xml_path = os.path.join(self.annot_dir, xml_name)
        
        # Load Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        
        # Load Annotation
        boxes = []
        labels = []
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.class_to_idx:
                    continue
                
                label = self.class_to_idx[name]
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        
        return img_tensor, target

def evaluate_map(model, data_loader, device):
    metric = MeanAveragePrecision()
    print("Running mAP evaluation...")
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            preds = model(images)
            
        metric.update(preds, targets)
        
    results = metric.compute()
    return results

def benchmark(data_dir=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Sub-system: {device}")

    # Load Model
    num_classes = len(CLASSES)
    model = get_resnet18_faster_rcnn(num_classes)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Warning: Model path {MODEL_PATH} not found.")

    model.to(device)
    model.eval()

    # --- mAP Evaluation ---
    if data_dir:
        try:
            dataset = PCBDataset(data_dir, CLASSES)
            if len(dataset) == 0:
                print("No images found in dataset.")
            else:
                loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
                map_results = evaluate_map(model, loader, device)
                print("\n--- mAP Results ---")
                print(f"mAP (IoU=0.50:0.95): {map_results['map']:.4f}")
                print(f"mAP_50 (IoU=0.50): {map_results['map_50']:.4f}")
                print(f"mAP_75 (IoU=0.75): {map_results['map_75']:.4f}")
                print("-" * 20)
        except Exception as e:
            print(f"Evaluation Failed: {e}")

    # --- FPS Benchmark ---
    # Load Image
    try:
        img_path = IMAGE_PATH
        # If dataset provided, pick first image from there for benchmark
        if data_dir:
             files = glob.glob(os.path.join(data_dir, 'images', '*')) 
             if files: img_path = files[0]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
            img_list = [img_tensor]

            # Warmup
            print("\nWarming up...")
            for _ in range(5):
                with torch.no_grad():
                    model(img_list)
            
            # Benchmark
            iterations = 50
            print(f"Running FPS benchmark ({iterations} iters)...")
            
            start_time = time.time()
            for _ in range(iterations):
                with torch.no_grad():
                    model(img_list)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_image = total_time / iterations
            fps = 1 / avg_time_per_image
            
            print(f"Average Inference Time: {avg_time_per_image*1000:.2f}ms")
            print(f"FPS: {fps:.2f}")
        else:
             print(f"Image for benchmark not found: {img_path}")
    except Exception as e:
        print(f"Benchmark Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PCB Defect Detection Model")
    parser.add_argument('--data_dir', type=str, help="Path to dataset directory (containing 'images' and 'annotations' folders) for mAP calculation")
    args = parser.parse_args()
    
    benchmark(args.data_dir)
