from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from datetime import datetime
import shutil
from tqdm import tqdm
import torch
from collections import Counter
import yaml
import matplotlib.font_manager as fm
import subprocess
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "logs/training_other_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}"
)

logger.add(
    lambda msg: print(msg),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}"
)

class OtherExperiment:
    def __init__(self, experiment_name, dataset_path, single_class=False):
        self.experiment_name = experiment_name
        self.dataset_path = Path(dataset_path)
        logger.info(f'self.dataset_path/n:{self.dataset_path}')
        self.single_class = single_class
        self.output_dir = Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model versions with environment names
        self.model_versions = {
            'yolo_fastestv2': {
                'name': 'Yolo-FastestV2',
                'repo': 'https://github.com/dog-qiuqiu/Yolo-FastestV2',
                'branch': 'main',
                'env_name': 'yolo_fastestv2_env',
                'python_version': '3.8'
            },
            'nanodet': {
                'name': 'NanoDet',
                'repo': 'https://github.com/RangiLyu/nanodet',
                'branch': 'main',
                'env_name': 'nanodet_env',
                'python_version': '3.8'
            },
            'picodet': {
                'name': 'PicoDet',
                'repo': 'https://github.com/PaddlePaddle/PaddleDetection',
                'branch': 'develop',
                'config': 'configs/picodet/picodet_s_320_coco.yml',
                'env_name': 'picodet_env',
                'python_version': '3.8'
            },
            'yolox_nano': {
                'name': 'YOLOX-Nano',
                'repo': 'https://github.com/Megvii-BaseDetection/YOLOX',
                'branch': 'main',
                'config': 'exps/yolox_nano.py',
                'env_name': 'yolox_nano_env',
                'python_version': '3.8'
            },
            'yolov5_nano': {
                'name': 'YOLOv5-Nano',
                'repo': 'https://github.com/ultralytics/yolov5',
                'branch': 'master',
                'config': 'models/yolov5n.yaml',
                'env_name': 'yolov5_nano_env',
                'python_version': '3.8'
            }
        }
        
        # Store current conda environment
        self.current_env = self._get_current_conda_env()
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
        # Initialize results storage
        self.results = {
            'mAP50': pd.DataFrame(),
            'mAP75': pd.DataFrame(),
            'mAP50_95': pd.DataFrame(),
            'train_speed': pd.DataFrame(),
            'inference_speed': pd.DataFrame()
        }
        self.model_info = {}
        
        # Get dataset statistics
        self.dataset_stats = self._get_dataset_stats()
        logger.info(f'self.dataset_stats/n:{self.dataset_stats}')
        
        # Setup matplotlib font
        self._setup_font()
        
        # Create single class dataset if needed
        self._create_single_class_dataset()
        
    def _setup_font(self):
        """Setup matplotlib font for Chinese characters"""
        try:
            font_paths = [
                'Fonts/SimSun.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/System/Library/Fonts/Arial.ttf',
                '/usr/share/fonts/TTF/SimSun.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            ]
            
            self.chinese_font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.chinese_font = fm.FontProperties(fname=font_path)
                    break
            
            if self.chinese_font is None:
                self.chinese_font = fm.FontProperties(family='sans-serif')
                
        except Exception as e:
            logger.warning(f"Font setup failed, using default font: {e}")
            self.chinese_font = fm.FontProperties(family='sans-serif')
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def _get_dataset_stats(self):
        """Get dataset statistics including class distribution"""
        stats = {
            'train': {'total': 0, 'classes': Counter()},
            'val': {'total': 0, 'classes': Counter()},
            'test': {'total': 0, 'classes': Counter()}
        }
        
        # Read dataset.yaml
        with open(self.dataset_path / "dataset.yaml", 'r') as f:
            dataset_config = yaml.safe_load(f)
            
        # Count images and classes
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / 'labels' / split
            if split_path.exists():
                for label_file in split_path.glob('*.txt'):
                    stats[split]['total'] += 1
                    with open(label_file, 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            stats[split]['classes'][class_id] += 1
                            
        return stats
    
    def _get_model_info(self, model_path):
        """Get model parameters and FLOPs"""
        try:
            # For Yolo-FastestV2 and NanoDet, we'll need to implement specific model info extraction
            # This is a placeholder that will need to be implemented based on each model's structure
            return 0, 0  # Return default values for now
        except Exception as e:
            logger.warning(f"Error calculating model info: {str(e)}")
            return 0, 0

    def _get_current_conda_env(self):
        """Get current conda environment name"""
        try:
            result = subprocess.run(
                ['conda', 'info', '--envs'],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if '*' in line:
                    return line.split()[0]
            return 'base'
        except Exception as e:
            logger.warning(f"Failed to get current conda environment: {str(e)}")
            return 'base'

    def _setup_conda_env(self, model_key):
        """Setup conda environment for a specific model"""
        model_info = self.model_versions[model_key]
        env_name = model_info['env_name']
        python_version = model_info['python_version']
        
        try:
            # Check if environment exists
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if env_name not in result.stdout:
                logger.info(f"Creating conda environment {env_name}...")
                # Create new environment
                subprocess.run([
                    'conda', 'create',
                    '-n', env_name,
                    f'python={python_version}',
                    '-y'
                ], check=True)
                
                # Install mlflow in the new environment
                subprocess.run([
                    'conda', 'run', '-n', env_name,
                    'pip', 'install', 'mlflow'
                ], check=True)
            
            return env_name
            
        except Exception as e:
            logger.error(f"Error setting up conda environment {env_name}: {str(e)}")
            raise

    def _activate_conda_env(self, env_name):
        """Activate conda environment"""
        try:
            # Use conda run to execute commands in the specified environment
            return ['conda', 'run', '-n', env_name]
        except Exception as e:
            logger.error(f"Error activating conda environment {env_name}: {str(e)}")
            raise

    def _setup_model_repo(self, model_key):
        """Setup model repository and dependencies"""
        model_info = self.model_versions[model_key]
        repo_dir = Path(f"models/{model_key}")
        env_name = model_info['env_name']
        
        if not repo_dir.exists():
            logger.info(f"Cloning {model_info['name']} repository...")
            subprocess.run([
                "git", "clone", 
                model_info['repo'], 
                str(repo_dir)
            ], check=True)
            
            # Checkout specific branch if needed
            if model_info.get('branch'):
                subprocess.run([
                    "git", "-C", str(repo_dir), 
                    "checkout", model_info['branch']
                ], check=True)
            
            # Install dependencies in the model's conda environment
            if (repo_dir / "requirements.txt").exists():
                logger.info(f"Installing dependencies for {model_info['name']}...")
                subprocess.run(
                    self._activate_conda_env(env_name) + [
                        "pip", "install", 
                        "-r", str(repo_dir / "requirements.txt")
                    ],
                    check=True
                )
        
        return repo_dir

    def train_and_evaluate(self):
        """Train and evaluate models"""
        # Log dataset statistics in a parent run
        with mlflow.start_run(run_name=f"{self.experiment_name}_overview"):
            mlflow.log_dict(self.dataset_stats, "dataset_stats.json")
        
        for model_key, model_info in tqdm(self.model_versions.items(), desc="Training models"):
            # Create a separate run for each model
            with mlflow.start_run(run_name=f"{self.experiment_name}_{model_key}", nested=True):
                try:
                    # Setup conda environment
                    env_name = self._setup_conda_env(model_key)
                    
                    # Setup model repository
                    repo_dir = self._setup_model_repo(model_key)
                    
                    # Get model info
                    params, flops = self._get_model_info(repo_dir)
                    
                    # Log model info
                    mlflow.log_params({
                        "model_name": model_info['name'],
                        "parameters": params,
                        "flops": flops
                    })
                    
                    # Train model using the model's conda environment
                    if model_key == 'yolo_fastestv2':
                        self._train_yolo_fastestv2(repo_dir, env_name)
                    elif model_key == 'nanodet':
                        self._train_nanodet(repo_dir, env_name)
                    elif model_key == 'picodet':
                        self._train_picodet(repo_dir, env_name)
                    elif model_key == 'yolox_nano':
                        self._train_yolox_nano(repo_dir, env_name)
                    elif model_key == 'yolov5_nano':
                        self._train_yolov5_nano(repo_dir, env_name)
                    
                except Exception as e:
                    logger.error(f"Error training {model_info['name']}: {str(e)}")
                    continue
        
        # Create and save heatmaps
        self._create_heatmaps()

    def _train_yolo_fastestv2(self, repo_dir, env_name):
        """Train Yolo-FastestV2 model"""
        try:
            # Change to the model directory
            os.chdir(repo_dir)
            
            # Copy existing dataset.yaml
            shutil.copy2(self.dataset_path / "dataset.yaml", "dataset.yaml")
            
            # Train model using the model's conda environment
            logger.info(f"Training Yolo-FastestV2 model...")
            results = subprocess.run(
                self._activate_conda_env(env_name) + [
                    'python', 'train.py',
                    '--data', 'dataset.yaml',
                    '--epochs', '200',
                    '--batch-size', '4',
                    '--img-size', '320',
                    '--device', '6,7',
                    '--workers', '8',
                    '--project', str(self.output_dir),
                    '--name', 'yolo_fastestv2_training',
                    '--exist-ok',
                    '--pretrained',
                    '--optimizer', 'AdamW',
                    '--lr0', '0.001',
                    '--lrf', '0.1',
                    '--momentum', '0.9',
                    '--weight-decay', '0.0005',
                    '--warmup-epochs', '3',
                    '--warmup-momentum', '0.8',
                    '--warmup-bias-lr', '0.1',
                    '--box', '7.5',
                    '--cls', '0.5',
                    '--dfl', '1.5',
                    '--close-mosaic', '10',
                    '--hsv-h', '0.015',
                    '--hsv-s', '0.7',
                    '--hsv-v', '0.4',
                    '--degrees', '0.0',
                    '--translate', '0.1',
                    '--scale', '0.5',
                    '--shear', '0.0',
                    '--perspective', '0.0',
                    '--flipud', '0.0',
                    '--fliplr', '0.5',
                    '--mosaic', '1.0',
                    '--mixup', '0.0',
                    '--copy-paste', '0.0',
                    '--save-period', '10'
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log training output
            logger.info(results.stdout)
            if results.stderr:
                logger.warning(results.stderr)
            
            # Evaluate every 10 epochs
            for epoch in range(10, 201, 10):
                model_path = self.output_dir / 'yolo_fastestv2_training' / f'weights/epoch{epoch}.pt'
                if model_path.exists():
                    eval_results = subprocess.run(
                        self._activate_conda_env(env_name) + [
                            'python', 'test.py',
                            '--data', 'dataset.yaml',
                            '--weights', str(model_path),
                            '--task', 'test'
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse and log metrics
                    metrics = self._parse_training_metrics(eval_results.stdout)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Update results dataframe
                    self._update_results('yolo_fastestv2', metrics, epoch)
            
        except Exception as e:
            logger.error(f"Error training Yolo-FastestV2: {str(e)}")
            raise
        finally:
            # Change back to original directory
            os.chdir(self.output_dir.parent)

    def _train_nanodet(self, repo_dir, env_name):
        """Train NanoDet model"""
        try:
            # Change to the model directory
            os.chdir(repo_dir)
            
            # Copy existing dataset.yaml
            shutil.copy2(self.dataset_path / "dataset.yaml", "dataset.yaml")
            
            # Train model using the model's conda environment
            logger.info(f"Training NanoDet model...")
            results = subprocess.run(
                self._activate_conda_env(env_name) + [
                    'python', 'tools/train.py',
                    'config/nanodet-m.yml',  # Using nano model configuration
                    '--data', 'dataset.yaml',
                    '--epochs', '200',
                    '--batch-size', '4',
                    '--img-size', '320',
                    '--device', '6,7',
                    '--workers', '8',
                    '--project', str(self.output_dir),
                    '--name', 'nanodet_training',
                    '--exist-ok',
                    '--pretrained',
                    '--optimizer', 'AdamW',
                    '--lr', '0.001',
                    '--weight-decay', '0.0005',
                    '--warmup-epochs', '3',
                    '--warmup-momentum', '0.8',
                    '--warmup-bias-lr', '0.1',
                    '--box-loss-weight', '7.5',
                    '--cls-loss-weight', '0.5',
                    '--close-mosaic', '10',
                    '--hsv-h', '0.015',
                    '--hsv-s', '0.7',
                    '--hsv-v', '0.4',
                    '--degrees', '0.0',
                    '--translate', '0.1',
                    '--scale', '0.5',
                    '--shear', '0.0',
                    '--perspective', '0.0',
                    '--flipud', '0.0',
                    '--fliplr', '0.5',
                    '--mosaic', '1.0',
                    '--mixup', '0.0',
                    '--copy-paste', '0.0',
                    '--save-period', '10'  # Save model every 10 epochs
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log training output
            logger.info(results.stdout)
            if results.stderr:
                logger.warning(results.stderr)
            
            # Evaluate every 10 epochs
            for epoch in range(10, 201, 10):
                model_path = self.output_dir / 'nanodet_training' / f'weights/epoch{epoch}.pt'
                if model_path.exists():
                    eval_results = subprocess.run(
                        self._activate_conda_env(env_name) + [
                            'python', 'tools/test.py',
                            '--data', 'dataset.yaml',
                            '--weights', str(model_path),
                            '--task', 'test'
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse and log metrics
                    metrics = self._parse_training_metrics(eval_results.stdout)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Update results dataframe
                    self._update_results('nanodet', metrics, epoch)
            
        except Exception as e:
            logger.error(f"Error training NanoDet: {str(e)}")
            raise
        finally:
            # Change back to original directory
            os.chdir(self.output_dir.parent)

    def _train_picodet(self, repo_dir, env_name):
        """Train PicoDet model"""
        try:
            # Change to the model directory
            os.chdir(repo_dir)
            
            # Copy existing dataset.yaml
            shutil.copy2(self.dataset_path / "dataset.yaml", "dataset.yaml")
            
            # Train model using the model's conda environment
            logger.info(f"Training PicoDet model...")
            results = subprocess.run(
                self._activate_conda_env(env_name) + [
                    'python', 'tools/train.py',
                    '--config', 'configs/picodet/picodet_s_320_coco.yml',
                    '--dataset_dir', str(self.dataset_path),
                    '--epochs', '200',
                    '--batch_size', '4',
                    '--img_size', '320',
                    '--device', '6,7',
                    '--workers', '8',
                    '--output_dir', str(self.output_dir / 'picodet_training'),
                    '--pretrained_weights', 'COCO',
                    '--optimizer', 'SGD',
                    '--lr', '0.001',
                    '--weight_decay', '0.0005',
                    '--warmup_epochs', '3',
                    '--warmup_momentum', '0.8',
                    '--warmup_bias_lr', '0.1',
                    '--box_loss_weight', '7.5',
                    '--cls_loss_weight', '0.5',
                    '--save_interval', '10'  # Save model every 10 epochs
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log training output
            logger.info(results.stdout)
            if results.stderr:
                logger.warning(results.stderr)
            
            # Evaluate every 10 epochs
            for epoch in range(10, 201, 10):
                model_path = self.output_dir / 'picodet_training' / f'model_epoch_{epoch}.pdparams'
                if model_path.exists():
                    eval_results = subprocess.run(
                        self._activate_conda_env(env_name) + [
                            'python', 'tools/eval.py',
                            '--config', 'configs/picodet/picodet_s_320_coco.yml',
                            '--weights', str(model_path),
                            '--dataset_dir', str(self.dataset_path)
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse and log metrics
                    metrics = self._parse_training_metrics(eval_results.stdout)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Update results dataframe
                    self._update_results('picodet', metrics, epoch)
            
        except Exception as e:
            logger.error(f"Error training PicoDet: {str(e)}")
            raise
        finally:
            # Change back to original directory
            os.chdir(self.output_dir.parent)

    def _train_yolox_nano(self, repo_dir, env_name):
        """Train YOLOX-Nano model"""
        try:
            # Change to the model directory
            os.chdir(repo_dir)
            
            # Copy existing dataset.yaml
            shutil.copy2(self.dataset_path / "dataset.yaml", "dataset.yaml")
            
            # Train model using the model's conda environment
            logger.info(f"Training YOLOX-Nano model...")
            results = subprocess.run(
                self._activate_conda_env(env_name) + [
                    'python', 'tools/train.py',
                    '-f', 'exps/yolox_nano.py',
                    '-d', '6,7',
                    '-b', '4',
                    '--epochs', '200',
                    '--img-size', '320',
                    '--data', 'dataset.yaml',
                    '--output_dir', str(self.output_dir / 'yolox_nano_training'),
                    '--pretrained',
                    '--optimizer', 'SGD',
                    '--lr', '0.001',
                    '--weight_decay', '0.0005',
                    '--warmup_epochs', '3',
                    '--warmup_momentum', '0.8',
                    '--warmup_bias_lr', '0.1',
                    '--save_interval', '10'  # Save model every 10 epochs
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log training output
            logger.info(results.stdout)
            if results.stderr:
                logger.warning(results.stderr)
            
            # Evaluate every 10 epochs
            for epoch in range(10, 201, 10):
                model_path = self.output_dir / 'yolox_nano_training' / f'epoch_{epoch}_ckpt.pth'
                if model_path.exists():
                    eval_results = subprocess.run(
                        self._activate_conda_env(env_name) + [
                            'python', 'tools/eval.py',
                            '-f', 'exps/yolox_nano.py',
                            '-d', '6,7',
                            '--data', 'dataset.yaml',
                            '--weights', str(model_path)
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse and log metrics
                    metrics = self._parse_training_metrics(eval_results.stdout)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Update results dataframe
                    self._update_results('yolox_nano', metrics, epoch)
            
        except Exception as e:
            logger.error(f"Error training YOLOX-Nano: {str(e)}")
            raise
        finally:
            # Change back to original directory
            os.chdir(self.output_dir.parent)

    def _train_yolov5_nano(self, repo_dir, env_name):
        """Train YOLOv5-Nano model"""
        try:
            # Change to the model directory
            os.chdir(repo_dir)
            
            # Copy existing dataset.yaml
            shutil.copy2(self.dataset_path / "dataset.yaml", "dataset.yaml")
            
            # Train model using the model's conda environment
            logger.info(f"Training YOLOv5-Nano model...")
            results = subprocess.run(
                self._activate_conda_env(env_name) + [
                    'python', 'train.py',
                    '--data', 'dataset.yaml',
                    '--cfg', 'models/yolov5n.yaml',
                    '--epochs', '200',
                    '--batch-size', '4',
                    '--img-size', '320',
                    '--device', '6,7',
                    '--workers', '8',
                    '--project', str(self.output_dir),
                    '--name', 'yolov5_nano_training',
                    '--exist-ok',
                    '--pretrained',
                    '--optimizer', 'AdamW',
                    '--lr0', '0.001',
                    '--lrf', '0.1',
                    '--momentum', '0.9',
                    '--weight-decay', '0.0005',
                    '--warmup-epochs', '3',
                    '--warmup-momentum', '0.8',
                    '--warmup-bias-lr', '0.1',
                    '--box', '7.5',
                    '--cls', '0.5',
                    '--dfl', '1.5',
                    '--close-mosaic', '10',
                    '--hsv-h', '0.015',
                    '--hsv-s', '0.7',
                    '--hsv-v', '0.4',
                    '--degrees', '0.0',
                    '--translate', '0.1',
                    '--scale', '0.5',
                    '--shear', '0.0',
                    '--perspective', '0.0',
                    '--flipud', '0.0',
                    '--fliplr', '0.5',
                    '--mosaic', '1.0',
                    '--mixup', '0.0',
                    '--copy-paste', '0.0',
                    '--save-period', '10'  # Save model every 10 epochs
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log training output
            logger.info(results.stdout)
            if results.stderr:
                logger.warning(results.stderr)
            
            # Evaluate every 10 epochs
            for epoch in range(10, 201, 10):
                model_path = self.output_dir / 'yolov5_nano_training' / f'weights/epoch{epoch}.pt'
                if model_path.exists():
                    eval_results = subprocess.run(
                        self._activate_conda_env(env_name) + [
                            'python', 'val.py',
                            '--data', 'dataset.yaml',
                            '--weights', str(model_path),
                            '--task', 'test'
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse and log metrics
                    metrics = self._parse_training_metrics(eval_results.stdout)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    # Update results dataframe
                    self._update_results('yolov5_nano', metrics, epoch)
            
        except Exception as e:
            logger.error(f"Error training YOLOv5-Nano: {str(e)}")
            raise
        finally:
            # Change back to original directory
            os.chdir(self.output_dir.parent)

    def _parse_training_metrics(self, output):
        """Parse training metrics from model output"""
        metrics = {}
        try:
            # Extract metrics from output
            metrics = {
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP50_95': 0.0,
                'train_speed': 0.0,
                'inference_speed': 0.0
            }
            
            # Try to extract metrics from output text
            for line in output.split('\n'):
                if 'mAP50' in line:
                    metrics['mAP50'] = float(line.split(':')[-1].strip())
                elif 'mAP75' in line:
                    metrics['mAP75'] = float(line.split(':')[-1].strip())
                elif 'mAP50-95' in line:
                    metrics['mAP50_95'] = float(line.split(':')[-1].strip())
                elif 'Speed' in line and 'train' in line.lower():
                    # Extract training speed (images/second)
                    speed_str = line.split('Speed:')[-1].strip().split()[0]
                    metrics['train_speed'] = float(speed_str)
                elif 'Speed' in line and 'inference' in line.lower():
                    # Extract inference speed (images/second)
                    speed_str = line.split('Speed:')[-1].strip().split()[0]
                    metrics['inference_speed'] = float(speed_str)
            
        except Exception as e:
            logger.warning(f"Error parsing metrics: {str(e)}")
        
        return metrics

    def _update_results(self, model_name, metrics, epoch=None):
        """Update results dataframe with new metrics"""
        for metric_name, value in metrics.items():
            if metric_name in self.results:
                if epoch is not None:
                    # For epoch-wise results
                    if model_name not in self.results[metric_name].index:
                        self.results[metric_name].loc[model_name] = pd.Series(index=range(10, 201, 10))
                    self.results[metric_name].loc[model_name, epoch] = value
                else:
                    # For final results
                    if model_name not in self.results[metric_name].index:
                        self.results[metric_name].loc[model_name] = value
                    else:
                        self.results[metric_name].loc[model_name] = value

    def _create_heatmaps(self):
        """Create heatmaps for each metric with model parameters and FLOPs"""
        for metric, df in self.results.items():
            # Skip if dataframe is empty
            if df.empty:
                logger.warning(f"No data available for {metric} heatmap")
                continue
                
            plt.figure(figsize=(20, 12))  # Larger figure size
            
            try:
                # Create heatmap
                sns.heatmap(
                    df,
                    annot=True,
                    fmt='.3f' if metric != 'inference_time' else '.2f',
                    cmap='YlOrRd',
                    cbar_kws={'label': f'{metric} (ms)' if metric == 'inference_time' else metric},
                    annot_kws={'size': 8}  # Annotation font size
                )
                
                # Get model info for y-axis labels
                y_labels = []
                for model_name in df.index:
                    try:
                        params, flops = self.model_info[model_name]["params"], self.model_info[model_name]["flops"]
                        # Format numbers for better readability
                        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.1f}K"
                        flops_str = f"{flops/1e9:.1f}G" if flops >= 1e9 else f"{flops/1e6:.1f}M"
                        model_line = f"{model_name}".center(40)
                        params_line = f"({params_str} params, {flops_str} FLOPs)".center(40)
                        y_labels.append(f"{model_line}\n{params_line}")
                    except Exception as e:
                        logger.warning(f"Failed to get model info for {model_name}: {str(e)}")
                        y_labels.append(model_name)
                
                # Title and axis labels
                plt.title(f"{self.experiment_name} - {metric}", fontproperties=self.chinese_font, fontsize=12)
                plt.xlabel("Epochs", fontproperties=self.chinese_font, fontsize=10)
                plt.ylabel("Models", fontproperties=self.chinese_font, fontsize=10)
                
                # Y-axis labels (model info)
                ax = plt.gca()
                ax.set_yticklabels(y_labels, fontproperties=self.chinese_font, verticalalignment='center', fontsize=7)
                
                # X-axis labels
                plt.xticks(rotation=45, fontsize=8)
                
                # Save heatmap
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / f"{metric}_heatmap.png",
                    dpi=300,  # High quality output
                    bbox_inches='tight',
                    format='png'
                )
                mlflow.log_artifact(str(self.output_dir / f"{metric}_heatmap.png"))
                
            except Exception as e:
                logger.error(f"Error creating heatmap for {metric}: {str(e)}")
            finally:
                plt.close()

    def _create_single_class_dataset(self):
        """Create single class dataset from the most frequent class with 7:2:1 split"""
        if not self.single_class:
            return
        
        # Check if single class dataset already exists and is valid
        single_class_dir = self.dataset_path.parent / "single_class_dataset"
        dataset_yaml = single_class_dir / "dataset.yaml"
        
        if dataset_yaml.exists():
            # Verify dataset.yaml and directories
            try:
                with open(dataset_yaml, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                
                # Check if all required directories exist
                paths_valid = True
                for split in ['train', 'val', 'test']:
                    split_path = single_class_dir / dataset_config[split]
                    if not split_path.exists():
                        paths_valid = False
                        logger.warning(f"Split directory not found: {split_path}")
                        break
                
                if paths_valid:
                    logger.info("Single class dataset already exists and is valid, skipping creation")
                    self.dataset_path = single_class_dir
                    return
                else:
                    logger.warning("Single class dataset exists but is invalid, recreating...")
            except Exception as e:
                logger.warning(f"Error validating existing dataset: {str(e)}, recreating...")
        
        # Read original dataset.yaml to get class names
        original_yaml_path = self.dataset_path / "dataset.yaml"
        with open(original_yaml_path, 'r') as f:
            original_config = yaml.safe_load(f)
            original_names = original_config.get('names', {})
        
        # Find most frequent class
        all_classes = Counter()
        for split in ['train', 'val', 'test']:
            all_classes.update(self.dataset_stats[split]['classes'])
        most_frequent_class = all_classes.most_common(1)[0][0]
        
        logger.info(f"Creating single class dataset for class {most_frequent_class} ({original_names.get(most_frequent_class, f'class_{most_frequent_class}')})")
        
        # Create new dataset directory
        single_class_dir.mkdir(exist_ok=True)
        
        # Collect all images and labels for the target class
        all_images = []
        all_labels = []
        
        for split in ['train', 'val', 'test']:
            for label_file in (self.dataset_path / 'labels' / split).glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = [line for line in f if int(line.split()[0]) == most_frequent_class]
                    
                if lines:  # If file contains the target class
                    img_file = self.dataset_path / 'images' / split / (label_file.stem + '.jpg')
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
            split_dir = single_class_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            for img_file, (label_file, lines) in zip(images, labels):
                # Copy image
                shutil.copy2(img_file, split_dir / img_file.name)
                
                # Write filtered labels
                with open(split_dir / label_file.name, 'w') as f:
                    f.writelines(lines)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(single_class_dir),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': {0: original_names.get(most_frequent_class, f'class_{most_frequent_class}')}
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f)
        
        # Update dataset path
        self.dataset_path = single_class_dir
        logger.info(f'Changed dataset path to: {self.dataset_path}')
        
        # Log split statistics
        logger.info(f"Single class dataset created with split sizes:")
        logger.info(f"Train: {len(splits['train'][0])} images")
        logger.info(f"Val: {len(splits['val'][0])} images")
        logger.info(f"Test: {len(splits['test'][0])} images")

def main():
    # Create experiments
    experiments = [
        OtherExperiment("other_multiclass_detect", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset"),
        OtherExperiment("other_singleclass_detect", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset", single_class=True),
        OtherExperiment("other_overfit_experiment", "/nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/label_data_dataset")
    ]
    
    # Run experiments
    for exp in experiments:
        try:
            logger.info(f"Starting experiment: {exp.experiment_name}")
            exp.train_and_evaluate()
            logger.info(f"Completed experiment: {exp.experiment_name}")
        except Exception as e:
            logger.error(f"Error in experiment {exp.experiment_name}: {str(e)}")
        finally:
            # Ensure MLflow run is ended
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error ending MLflow run: {str(e)}")

if __name__ == "__main__":
    main() 