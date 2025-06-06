import os
import shutil
import argparse
from pathlib import Path
import yaml
from loguru import logger
from tqdm import tqdm

def ensure_dir(dir_path):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(dir_path, exist_ok=True)

def copy_and_convert_dataset(input_dir: str, output_dir: str = None):
    # Define paths
    source_root = Path(input_dir)
    if output_dir is None:
        output_dir = str(source_root) + "_yolo"
    target_root = Path(output_dir)
    
    # Remove output directory if it exists
    if target_root.exists():
        logger.info(f"Removing existing output directory: {target_root}")
        shutil.rmtree(target_root)
    
    logger.info(f"Converting dataset from {source_root} to {target_root}")
    
    # Create necessary directories
    for split in ['train', 'val', 'test']:
        ensure_dir(target_root / split)
        logger.info(f"Created directory: {target_root / split}")
    
    # Copy category names
    yaml_path = source_root / "dataset.yaml"
    logger.info(f"Reading dataset config from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Write category names
    category_names_path = target_root / "category.names"
    logger.info(f"Writing category names to: {category_names_path}")
    with open(category_names_path, 'w') as f:
        for class_name in dataset_config['names'].values():
            f.write(f"{class_name}\n")
    
    # Process each split
    for split in tqdm(['train', 'val', 'test'],desc="Processing splits",colour="green",leave=False):
        # Get image paths
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        
        logger.info(f"Processing {split} split...")
        
        # Create list file for this split
        list_file_path = target_root / f"{split}.txt"
        logger.info(f"Creating list file: {list_file_path}")
        
        with open(list_file_path, 'w') as f:
            image_count = 0
            for img_file in tqdm(sorted(image_dir.glob("*.jpg")),desc=f"Processing {split} split",colour="yellow",leave=True):
                # Copy image
                target_img_path = target_root / split / img_file.name
                shutil.copy2(img_file, target_img_path)
                
                # Write image path to list file
                write_path=target_root.joinpath(split,img_file.name).as_posix()
                f.write(f"{write_path}\n")
                
                # Copy corresponding label
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    target_label_path = target_root / split / label_file.name
                    shutil.copy2(label_file, target_label_path)
                    image_count += 1
                else:
                    logger.warning(f"No label file found for {img_file.name}")
            
            logger.info(f"Processed {image_count} images for {split} split")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to YOLO format')
    parser.add_argument('--input_dir', type=str,required=True, help='Input directory containing the dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory (default: input_dir + "_yolo")')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    try:
        copy_and_convert_dataset(args.input_dir, args.output_dir)
        logger.success("Dataset conversion completed successfully!")
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main() 