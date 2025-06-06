import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
from loguru import logger

def get_dataset_stats(dataset_path):
    """Get dataset statistics including class distribution"""
    stats = {
        'train': {'total': 0, 'classes': Counter()},
        'val': {'total': 0, 'classes': Counter()},
        'test': {'total': 0, 'classes': Counter()}
    }
    
    # Read dataset.yaml
    with open(Path(dataset_path) / "dataset.yaml", 'r') as f:
        dataset_config = yaml.safe_load(f)
        
    # Count images and classes
    for split in ['train', 'val', 'test']:
        split_path = Path(dataset_path) / 'labels' / split
        if split_path.exists():
            for label_file in split_path.glob('*.txt'):
                stats[split]['total'] += 1
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        stats[split]['classes'][class_id] += 1
                        
    return stats

def create_single_class_dataset(input_dir, output_dir=None):
    """
    Create single class dataset from the most frequent class with 7:2:1 split
    
    Args:
        input_dir (str): Path to the input dataset directory
        output_dir (str, optional): Path to the output directory. If None, will use input_dir/single_class_dataset
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / "single_class_dataset"
    else:
        output_dir = Path(output_dir)
    
    dataset_yaml = output_dir / "dataset.yaml"
    
    # If output directory exists, remove it
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Read original dataset.yaml to get class names
    original_yaml_path = input_dir / "dataset.yaml"
    with open(original_yaml_path, 'r') as f:
        original_config = yaml.safe_load(f)
        original_names = original_config.get('names', {})
    
    # Get dataset statistics
    dataset_stats = get_dataset_stats(input_dir)
    
    # Find most frequent class
    all_classes = Counter()
    for split in ['train', 'val', 'test']:
        all_classes.update(dataset_stats[split]['classes'])
    most_frequent_class = all_classes.most_common(1)[0][0]
    
    logger.info(f"Creating single class dataset for class {most_frequent_class} ({original_names.get(most_frequent_class, f'class_{most_frequent_class}')})")
    
    # Create new dataset directory
    output_dir.mkdir(exist_ok=True)
    
    # Collect all images and labels for the target class
    all_images = []
    all_labels = []
    
    for split in ['train', 'val', 'test']:
        for label_file in (input_dir / 'labels' / split).glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = [line for line in f if int(line.split()[0]) == most_frequent_class]
                
            if lines:  # If file contains the target class
                img_file = input_dir / 'images' / split / (label_file.stem + '.jpg')
                if img_file.exists():
                    all_images.append(img_file)
                    # Convert class ID to 0 for single class
                    converted_lines = [' '.join(['0'] + line.split()[1:]) for line in lines]
                    all_labels.append((label_file, converted_lines))
    
    # Shuffle the data
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Calculate split sizes
    total_size = len(all_images)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.2)
    
    # Split the data
    splits = {
        'train': (all_images[:train_size], all_labels[:train_size]),
        'val': (all_images[train_size:train_size+val_size], all_labels[train_size:train_size+val_size]),
        'test': (all_images[train_size+val_size:], all_labels[train_size+val_size:])
    }
    
    # Create and populate split directories
    for split_name, (images, labels) in splits.items():
        split_image_dir = output_dir /'images'/ split_name
        split_label_dir=output_dir /'labels'/ split_name
        split_image_dir.mkdir(parents=True,exist_ok=True)
        split_label_dir.mkdir(parents=True,exist_ok=True)
        
        for img_file, (label_file, lines) in zip(images, labels):
            # Copy image
            shutil.copy2(img_file, split_image_dir / img_file.name)
            
            # Write filtered labels
            with open(split_label_dir / label_file.name, 'w') as f:
                f.writelines(lines)
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: original_names.get(most_frequent_class, f'class_{most_frequent_class}')}
    }
    
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Log split statistics
    logger.info(f"Single class dataset created with split sizes:")
    logger.info(f"Train: {len(splits['train'][0])} images")
    logger.info(f"Val: {len(splits['val'][0])} images")
    logger.info(f"Test: {len(splits['test'][0])} images")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create single class dataset from multi-class dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input dataset directory')
    parser.add_argument('--output_dir', type=str, help='Path to output directory (optional)')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    create_single_class_dataset(args.input_dir, args.output_dir) 