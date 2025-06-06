#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO format to COCO format')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing YOLO format data')
    parser.add_argument('--output_dir', type=str, help='Output directory for COCO format data')
    parser.add_argument('--debug', action='store_true', help='If set, sample 5 images and draw annotations for debugging')
    return parser.parse_args()

def create_coco_structure():
    """Create basic COCO structure"""
    return {
        "info": {
            "description": "Converted from YOLO format",
            "version": "1.0",
            "year": 2024,
            "contributor": "YOLO2COCO Converter",
            "date_created": "2024"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

def read_yolo_labels(label_path, img_width, img_height):
    """Read YOLO format labels and convert to COCO format"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            try:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert YOLO format to COCO format
                x_min = (x_center - width/2) * img_width
                y_min = (y_center - height/2) * img_height
                width = width * img_width
                height = height * img_height
                
                annotation = {
                    "id": len(annotations) + 1,
                    "image_id": 0,  # Will be set later
                    "category_id": int(class_id) + 1,  # COCO uses 1-based indexing
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "segmentation": [],
                    "iscrowd": 0
                }
                annotations.append(annotation)
            except Exception as e:
                logger.error(f"Error processing line in {label_path}: {line.strip()}")
                logger.error(f"Error details: {str(e)}")
                continue
    
    return annotations

def draw_coco_debug_images(coco_data, output_dir, num_samples=5):
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    if len(coco_data['images']) == 0:
        logger.warning('No images to debug!')
        return

    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    logger.info(f"Debug mode - Category mapping: {cat_id_to_name}")

    sampled_images = random.sample(coco_data['images'], min(num_samples, len(coco_data['images'])))
    anns_by_image = {}
    for ann in coco_data['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    for img_info in sampled_images:
        img_path = os.path.join(output_dir, 'images', img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Debug: Could not read image: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        anns = anns_by_image.get(img_info['id'], [])
        logger.info(f"Debug: Processing image {img_info['file_name']} with {len(anns)} annotations")
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x, y, w, h = map(int, [x, y, w, h])
            
            # Draw bounding box
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Get category name
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            
            # Add category label
            ax.text(x, y-5, f"{cat_name} (ID: {cat_id})", color='yellow', fontsize=10, 
                   bbox=dict(facecolor='red', alpha=0.5))
            
            logger.debug(f"Debug: Drawing annotation - category={cat_name} (ID: {cat_id}), bbox=[{x}, {y}, {w}, {h}]")
        
        ax.set_axis_off()
        plt.tight_layout()
        debug_img_path = os.path.join(debug_dir, f"debug_{img_info['file_name']}")
        plt.savefig(debug_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logger.info(f"Debug image saved: {debug_img_path}")

def convert_yolo_to_coco(input_dir, output_dir, debug=False):
    """Convert YOLO format dataset to COCO format"""
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))
    
    # Read class names
    class_names = []
    category_file = os.path.join(input_dir, 'category.names')
    logger.info(f"Reading category names from: {category_file}")
    with open(category_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    logger.info(f"Found {len(class_names)} categories: {class_names}")
    
    # Create COCO structure
    coco_data = create_coco_structure()
    
    # Add categories
    for idx, name in enumerate(class_names):
        category = {
            "id": idx + 1,  # COCO uses 1-based indexing
            "name": name,
            "supercategory": "none"
        }
        coco_data["categories"].append(category)
        logger.info(f"Added category: {category}")
    
    # Process each split (train, val, test)
    splits = ['train', 'val', 'test']
    for split in tqdm(splits, desc="Processing splits", colour="green", leave=True):
        logger.info(f"Processing {split} split...")
        
        # Read image list
        split_file = os.path.join(input_dir, f'{split}.txt')
        if not os.path.exists(split_file):
            logger.warning(f"Split file {split_file} not found, skipping...")
            continue
            
        with open(split_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        # Process each image
        for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Converting {split} images", colour="yellow", leave=True)):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
                
            height, width = img.shape[:2]
            
            # Create image entry
            image_entry = {
                "id": len(coco_data["images"]) + 1,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "2024"
            }
            coco_data["images"].append(image_entry)
            
            # Process annotations
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            annotations = read_yolo_labels(label_path, width, height)
            
            # Log annotation details
            if annotations:
                logger.debug(f"Image {img_path} has {len(annotations)} annotations")
                for ann in annotations:
                    cat_id = ann['category_id']
                    cat_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == cat_id), f"Unknown_{cat_id}")
                    logger.debug(f"Annotation: category_id={cat_id}, category_name={cat_name}, bbox={ann['bbox']}")
            
            # Update image_id in annotations
            for ann in annotations:
                ann["image_id"] = image_entry["id"]
                coco_data["annotations"].append(ann)
            
            # Copy image to output directory
            dst_path = os.path.join(output_dir, 'images', os.path.basename(img_path))
            shutil.copy2(img_path, dst_path)
    
    # Save COCO format annotation file
    output_file = os.path.join(output_dir, 'annotations.json')
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"Conversion completed. Output saved to {output_dir}")
    logger.info(f"Total images: {len(coco_data['images'])}")
    logger.info(f"Total annotations: {len(coco_data['annotations'])}")
    logger.info(f"Total categories: {len(coco_data['categories'])}")
    
    # Print category mapping for verification
    logger.info("Category mapping:")
    for cat in coco_data['categories']:
        logger.info(f"ID {cat['id']}: {cat['name']}")

    if debug:
        logger.info("Debug mode enabled: Drawing sample images with annotations.")
        draw_coco_debug_images(coco_data, output_dir)

def main():
    args = parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = args.input_dir + "_merge_coco"
    
    logger.info(f"Converting YOLO dataset from {args.input_dir} to COCO format in {args.output_dir}")
    convert_yolo_to_coco(args.input_dir, args.output_dir, debug=args.debug)

if __name__ == "__main__":
    main() 