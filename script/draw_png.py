from train_yolo_experiments import YOLOExperiment
import pandas as pd
import numpy as np
from pathlib import Path
import os
from loguru import logger
from tqdm import tqdm
import argparse

def read_nanodet_results(base_path):
    """Read NanoDet evaluation results from eval_results.txt files"""
    results = {
        'mAP50': pd.DataFrame(),
        'mAP75': pd.DataFrame(),
        'mAP50_95': pd.DataFrame(),
        'train_speed': pd.DataFrame(),
        'inference_speed': pd.DataFrame()
    }
    
    # Get all test directories
    test_dirs = [d for d in Path(base_path).glob('test_*') if d.is_dir()]
    test_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    # Extract model name from base path
    model_name = Path(base_path).name.split('_')[0]
    logger.info(f"Model name extracted: {model_name}")

    for test_dir in tqdm(test_dirs,desc="Reading NanoDet results",colour='green'):
        logger.info(f"Reading {test_dir}")
        epoch = int(test_dir.name.split('_')[1])
        eval_file = test_dir / 'eval_results.txt'
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if 'mAP' in line:
                            results['mAP50_95'].loc[model_name, epoch] = float(line.split(':')[-1].strip())
                        elif 'AP_50' in line:
                            results['mAP50'].loc[model_name, epoch] = float(line.split(':')[-1].strip())
                        elif 'AP_75' in line:
                            results['mAP75'].loc[model_name, epoch] = float(line.split(':')[-1].strip())
            except Exception as e:
                logger.warning(f"Error reading {eval_file}: {str(e)}")
    
    return model_name,results



def create_nanodet_heatmaps(model_name,results, output_dir):
    """Create heatmaps for NanoDet results"""
    exp = YOLOExperiment("other_multiclass_detect", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset")
    exp.results = results
    exp.model_info = {model_name: {"params": 0.70e6, "flops": 0.67e9}}
    exp.output_dir = Path(output_dir)
    exp._create_heatmaps()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate heatmaps for NanoDet results.")
    parser.add_argument("--base_path", type=str, help="Base path for NanoDet results", required=True)
    args = parser.parse_args()

    # Use the base_path from arguments
    base_path = args.base_path
    
    # Read NanoDet results
    model_name,results = read_nanodet_results(base_path)
    
    # Create output directory
    # The output directory will be relative to the base_path for consistency
    output_dir = Path(base_path)  # Or some other relative path if preferred
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmaps
    create_nanodet_heatmaps(model_name,results, output_dir)
    
    logger.info(f"Heatmaps created successfully in {output_dir}!")

if __name__ == "__main__":
    main()