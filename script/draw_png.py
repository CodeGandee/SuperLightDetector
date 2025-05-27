from train_yolo_experiments import YOLOExperiment
import pandas as pd
import numpy as np
from pathlib import Path
import os
from loguru import logger
from tqdm import tqdm

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
                            results['mAP50_95'].loc['nanodet', epoch] = float(line.split(':')[-1].strip())
                        elif 'AP_50' in line:
                            results['mAP50'].loc['nanodet', epoch] = float(line.split(':')[-1].strip())
                        elif 'AP_75' in line:
                            results['mAP75'].loc['nanodet', epoch] = float(line.split(':')[-1].strip())
            except Exception as e:
                logger.warning(f"Error reading {eval_file}: {str(e)}")
    
    return results



def create_nanodet_heatmaps(results, output_dir):
    """Create heatmaps for NanoDet results"""
    exp = YOLOExperiment("other_multiclass_detect", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset")
    exp.results = results
    exp.model_info = {"nanodet": {"params": 1.17e6, "flops": 0.9e9}}
    exp.output_dir = Path(output_dir)
    exp._create_heatmaps()

def main():
    # Base path for NanoDet results
    base_path = "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/experiments/other_multiclass_detect/nanodet_training-lr001-ratio001"
    
    # Read NanoDet results
    results = read_nanodet_results(base_path)
    
    # Create output directory
    output_dir = Path("experiments/other_multiclass_detect/nanodet_training-lr001-ratio001")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmaps
    create_nanodet_heatmaps(results, output_dir)
    
    logger.info("Heatmaps created successfully!")

if __name__ == "__main__":
    main()