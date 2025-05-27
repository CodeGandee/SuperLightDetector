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
    
    base_path = Path(base_path)
    model_names = []
    
    # Check if there are test_* directories directly under base_path
    test_dirs = [d for d in base_path.glob('test_*') if d.is_dir()]
    
    if test_dirs:
        # Case 1: test_* directories directly under base_path
        test_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        model_file_name = base_path.name
        model_main_name = model_file_name.split('_')[0]
        if '-' in model_file_name:
            model_flag_name = model_file_name.split('-')[1]
            model_name = model_main_name + '_' + model_flag_name
        else:
            model_name = model_main_name
        model_names.append(model_name)
        logger.info(f"Model name extracted: {model_name}")
        
        for test_dir in tqdm(test_dirs, desc="Reading NanoDet results", colour='green'):
            process_test_dir(test_dir, model_name, results)
    else:
        # Case 2: test_* directories under subdirectories
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        subdirs.sort(key=lambda x: x.name.split('_')[0])
        for subdir in subdirs:
            test_dirs = [d for d in subdir.glob('test_*') if d.is_dir()]
            if test_dirs:
                test_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
                model_main_name = subdir.name.split('_')[0]
                if '-' in subdir.name:
                    model_flag_name = subdir.name.split('-')[1]
                    model_name = model_main_name + '_' + model_flag_name
                else:
                    model_name = model_main_name
                model_names.append(model_name)
                logger.info(f"Processing subdirectory: {model_name}")
                
                for test_dir in tqdm(test_dirs, desc=f"Reading {model_name} results", colour='green'):
                    process_test_dir(test_dir, model_name, results)
    
    return model_names, results

def process_test_dir(test_dir, model_name, results):
    """Process a single test directory and update results"""
    # logger.info(f"Reading {test_dir}")
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

def create_nanodet_heatmaps(model_names, results, output_dir):
    """Create heatmaps for NanoDet results"""
    exp = YOLOExperiment("other_multiclass_detect", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset")
    exp.results = results
    
    # Create model info dictionary for all models
    model_info = {}
    for model_name in model_names:
        if "nanodet" in model_name:
            model_info[model_name] = {"params": 1.17e6, "flops": 0.9e9}
        elif "picodet" in model_name:
            model_info[model_name] = {"params": 0.70e6, "flops": 0.67e9}
        else:
            raise ValueError(f"Model name {model_name} not supported")
    
    exp.model_info = model_info
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
    model_names, results = read_nanodet_results(base_path)
    
    # Create output directory
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmaps
    create_nanodet_heatmaps(model_names, results, output_dir)
    
    logger.info(f"Heatmaps created successfully in {output_dir}!")

if __name__ == "__main__":
    main()