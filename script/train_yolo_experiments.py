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

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "logs/training_{time}.log",
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

class YOLOExperiment:
    def __init__(self, experiment_name, dataset_path, single_class=False):
        self.experiment_name = experiment_name
        self.dataset_path = Path(dataset_path)
        logger.info(f'self.dataset_path/n:{self.dataset_path}')
        self.single_class = single_class
        self.output_dir = Path(f"experiments/{experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model versions (smallest models for each version)
        self.model_versions = [
            "yolo12n",    # YOLO12n
            "yolo11n",    # YOLOv11n 
            "yolov10n",   # YOLOv10n
            "yolov9t",    # YOLOv9t
            "yolov8n",    # YOLOv8n
            "yolov5nu",   # YOLOv5n ultralytics version
            "yolov3u" 
        ]
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
        # Initialize results storage
        self.results = {
            'mAP50': pd.DataFrame(),
            'mAP75': pd.DataFrame(),
            'mAP50_95': pd.DataFrame(),
            'inference_time': pd.DataFrame()
        }
        self.model_info = {}
        # Get dataset statistics
        self.dataset_stats = self._get_dataset_stats()
        logger.info(f'self.dataset_stats/n:{self.dataset_stats}')
        
        # Setup matplotlib font
        self._setup_font()

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
    
    def _get_model_info(self, model):
        """Get model parameters and FLOPs"""
        try:
            # Get total parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Calculate FLOPs
            # For YOLO models, we can estimate FLOPs based on model architecture
            # Using a rough estimation based on input size and model parameters
            input_size = 320  # Default input size
            flops = total_params * 2 * input_size * input_size / 1000  # Rough estimation
            
            return total_params, flops
        except Exception as e:
            logger.warning(f"Error calculating model info: {str(e)}")
            return 0, 0  # Return default values if calculation fails
    
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

    def train_and_evaluate(self):
        # Initialize results storage for this model
        self.results['mAP50'] = pd.DataFrame()
        self.results['mAP75'] = pd.DataFrame()
        self.results['mAP50_95'] = pd.DataFrame()
        self.results['inference_time'] = pd.DataFrame()
        self.model_info = {}
        """Train and evaluate models"""
        # Log dataset statistics in a parent run
        with mlflow.start_run(run_name=f"{self.experiment_name}_overview"):
            mlflow.log_dict(self.dataset_stats, "dataset_stats.json")
        
        for model_name in tqdm(self.model_versions, desc="Training models"):
            # Create a separate run for each model
            with mlflow.start_run(run_name=f"{self.experiment_name}_{model_name}", nested=True):
                try:
                    # Check if model is already trained
                    model_dir = self.output_dir / f"{model_name}_training"
                    last_model_path = model_dir / "weights/last.pt"
                    is_trained = last_model_path.exists()
                    # Load model
                    model = YOLO(f"./models/{model_name}.pt")
                    params, flops = self._get_model_info(model)
                    self.model_info[model_name] = {
                        "params": params,
                        "flops": flops
                    }
                    # Log model info
                    mlflow.log_params({
                        "model_name": model_name,
                        "parameters": params,
                        "flops": flops
                    })
                    
                    if not is_trained:
                        # Train model for 200 epochs with evaluation every 10 epochs
                        results = model.train(
                            data=str(self.dataset_path / "dataset.yaml"),
                            epochs=200,
                            imgsz=320,
                            batch=4,
                            device="2,3",
                            lr0=0.001,
                            lrf=0.1,
                            momentum=0.9,
                            weight_decay=0.0005,
                            warmup_epochs=3,
                            warmup_momentum=0.8,
                            warmup_bias_lr=0.1,
                            box=7.5,
                            cls=0.5,
                            dfl=1.5,
                            val=True,
                            save=True,
                            save_period=10,  # Save model every 10 epochs
                            plots=True,
                            verbose=False,
                            project=str(self.output_dir),
                            name=f"{model_name}_training",
                            exist_ok=True,
                            pretrained=True,
                            optimizer='AdamW',
                            close_mosaic=10,
                            resume=False,
                            amp=False,
                            fraction=1.0,
                            profile=False,
                            # Data augmentation parameters
                            hsv_h=0.015,
                            hsv_s=0.7,
                            hsv_v=0.4,
                            degrees=0.0,
                            translate=0.1,
                            scale=0.5,
                            shear=0.0,
                            perspective=0.0,
                            flipud=0.0,
                            fliplr=0.5,
                            mosaic=1.0,
                            mixup=0.0,
                            copy_paste=0.0
                        )
                    else:
                        logger.info(f"Model {model_name} already trained, skipping training step")
                    
                    # Evaluate at every 10 epochs
                    for epoch in range(10, 201, 10):
                        # Load the saved model for this epoch
                        epoch_model_path = self.output_dir / f"{model_name}_training" / f"weights/epoch{epoch}.pt"
                        if epoch == 200:
                            epoch_model_path = self.output_dir / f"{model_name}_training" / f"weights/last.pt"
                        logger.info(f"Epoch model path: {epoch_model_path}")
                        if epoch_model_path.exists():
                            epoch_model = YOLO(str(epoch_model_path))
                            
                            # Evaluate on test set with save=False to prevent creating val folders
                            metrics = epoch_model.val(
                                data=str(self.dataset_path / "dataset.yaml"),
                                split='test',
                                plots=True,
                                save=False  # Prevent creating val folders
                            )

                            # Log metrics
                            mlflow.log_metrics({
                                "mAP50": metrics.box.map50,
                                "mAP75": metrics.box.map75,
                                "mAP50_95": metrics.box.map,
                                "inference_time": metrics.speed['inference']
                            }, step=epoch)
                            
                            # Store results
                            self.results['mAP50'].loc[model_name, epoch] = metrics.box.map50
                            self.results['mAP75'].loc[model_name, epoch] = metrics.box.map75
                            self.results['mAP50_95'].loc[model_name, epoch] = metrics.box.map
                            self.results['inference_time'].loc[model_name, epoch] = metrics.speed['inference']
                            
                            logger.info(f"Epoch {epoch} evaluation for {model_name}:")
                            logger.info(f"mAP50: {metrics.box.map50:.3f}")
                            logger.info(f"mAP75: {metrics.box.map75:.3f}")
                            logger.info(f"mAP50_95: {metrics.box.map:.3f}")
                            logger.info(f"Inference time: {metrics.speed['inference']:.2f}s")
                        else:
                            logger.warning(f"Model checkpoint not found for epoch {epoch} and epoch_model_path: {epoch_model_path}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
        
        # Create and save heatmaps
        self._create_heatmaps()

    def _create_heatmaps(self):
        """Create heatmaps for each metric with model parameters and FLOPs"""
        
        # Determine if it's a single-model case
        all_model_names = set()
        first_non_empty_df_for_columns = None
        for df_check in self.results.values():
            if not df_check.empty:
                all_model_names.update(df_check.index.tolist())
                if first_non_empty_df_for_columns is None:
                    first_non_empty_df_for_columns = df_check
        
        is_single_model_case = len(all_model_names) == 1
        
        if is_single_model_case and first_non_empty_df_for_columns is not None:
            single_model_name = list(all_model_names)[0]
            logger.info(f"Single model case detected: {single_model_name}. Generating combined metrics heatmap.")

            data_for_heatmap = {}
            common_epochs = first_non_empty_df_for_columns.columns.tolist()

            for metric, df_metric_data in self.results.items():
                if not df_metric_data.empty and single_model_name in df_metric_data.index:
                    # Ensure the series is aligned with common_epochs
                    metric_series = df_metric_data.loc[single_model_name].reindex(common_epochs)
                    data_for_heatmap[metric] = metric_series
            
            if not data_for_heatmap:
                logger.warning(f"No data found for single model {single_model_name} to create combined heatmap.")
                # Fallback to original behavior or simply return
                # For now, let's try to make sure the original loop doesn't run if this was triggered.
                # However, if data_for_heatmap is empty, maybe we should let the original loop try.
                # Let's proceed assuming if is_single_model_case is true, this is the desired plot.
                # If data_for_heatmap is empty, then an empty plot or error will occur below.
                # This needs careful handling if no data means use old method.
                # For now, if it's a single model name, it *must* be plotted this way or not at all.
                if not data_for_heatmap:
                    logger.warning(f"No metric data for {single_model_name}, cannot create combined heatmap.")
                    return


            combined_heatmap_df = pd.DataFrame(data_for_heatmap) # Metrics are columns, epochs are index
            if combined_heatmap_df.empty:
                logger.warning(f"Combined DataFrame for {single_model_name} is empty. Skipping heatmap.")
                return
            combined_heatmap_df = combined_heatmap_df.T # Metrics are rows, epochs are columns

            if combined_heatmap_df.empty: # Check again after transpose
                logger.warning(f"Transposed combined DataFrame for {single_model_name} is empty. Skipping heatmap.")
                return

            num_epochs = len(common_epochs)
            num_metrics = len(combined_heatmap_df.index)
            fig_width = max(15, num_epochs * 0.6)
            fig_height = max(8, num_metrics * 1.0)

            plt.figure(figsize=(fig_width, fig_height))
            
            try:
                sns.heatmap(
                    combined_heatmap_df,
                    annot=True,
                    fmt='.3f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Metric Value'},
                    annot_kws={'size': 8}
                )
                
                model_params_str, model_flops_str = "", ""
                if single_model_name in self.model_info:
                    params, flops = self.model_info[single_model_name]["params"], self.model_info[single_model_name]["flops"]
                    model_params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.1f}K"
                    model_flops_str = f"{flops/1e9:.1f}G" if flops >= 1e9 else f"{flops/1e9:.1f}G"
                
                title = f"{self.experiment_name} - Metrics for {single_model_name}"
                if model_params_str and model_flops_str:
                    title += f" ({model_params_str} params, {model_flops_str} FLOPs)"
                
                plt.title(title, fontproperties=self.chinese_font, fontsize=12)
                plt.xlabel("Epochs", fontproperties=self.chinese_font, fontsize=10)
                plt.ylabel("Metrics", fontproperties=self.chinese_font, fontsize=10)
                
                ax = plt.gca()
                ax.set_yticklabels(combined_heatmap_df.index, fontproperties=self.chinese_font, verticalalignment='center', fontsize=9, rotation=0)
                plt.xticks(rotation=45, fontsize=8)
                
                plt.tight_layout()
                save_path = self.output_dir / f"{single_model_name}_combined_metrics_heatmap.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
                if self.mlflow_run: # Check if mlflow run is active
                    mlflow.log_artifact(str(save_path))
                logger.info(f"Saved combined heatmap for {single_model_name} to {save_path}")
                
            except Exception as e:
                logger.error(f"Error creating combined heatmap for {single_model_name}: {str(e)}")
            finally:
                plt.close()
            return # Important: return after handling the single model case

        # Original multi-model plotting logic
        for metric, df in self.results.items():
            # Skip if dataframe is empty
            if df.empty:
                logger.warning(f"No data available for {metric} heatmap")
                continue
                
            plt.figure(figsize=(20, 12))  # 更大的图像尺寸
            
            try:
                # Create heatmap
                sns.heatmap(
                    df,
                    annot=True,
                    fmt='.3f' if metric != 'inference_time' else '.2f',
                    cmap='YlOrRd',
                    cbar_kws={'label': f'{metric} (ms)' if metric == 'inference_time' else metric},
                    annot_kws={'size': 8}  # 数值标注字体大小
                )
                
                # Get model info for y-axis labels
                y_labels = []
                for model_name in df.index:
                    try:
                        params, flops = self.model_info[model_name]["params"], self.model_info[model_name]["flops"]
                        # Format numbers for better readability
                        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.1f}K"
                        flops_str = f"{flops/1e9:.1f}G" if flops >= 1e9 else f"{flops/1e9:.1f}G"
                        model_line = f"{model_name}".center(40)
                        params_line = f"({params_str} params, {flops_str} FLOPs)".center(40)
                        y_labels.append(f"{model_line}\n{params_line}")
                    except Exception as e:
                        logger.warning(f"Failed to get model info for {model_name}: {str(e)}")
                        y_labels.append(model_name)
                
                # 标题和轴标签
                plt.title(f"{self.experiment_name} - {metric}", fontproperties=self.chinese_font, fontsize=12)
                plt.xlabel("Epochs", fontproperties=self.chinese_font, fontsize=10)
                plt.ylabel("Models", fontproperties=self.chinese_font, fontsize=10)
                
                # Y轴标签（模型信息）
                ax = plt.gca()
                ax.set_yticklabels(y_labels, fontproperties=self.chinese_font, verticalalignment='center', fontsize=7)
                
                # X轴标签
                plt.xticks(rotation=45, fontsize=8)
                
                # Save heatmap
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / f"{metric}_heatmap.png",
                    dpi=300,  # High quality output
                    bbox_inches='tight',
                    format='png'
                )
                if self.mlflow_run: # Check if mlflow run is active
                    mlflow.log_artifact(str(self.output_dir / f"{metric}_heatmap.png"))
                
            except Exception as e:
                logger.error(f"Error creating heatmap for {metric}: {str(e)}")
            finally:
                plt.close()

    def _check_mlflow_connection(self):
        """检查MLflow连接状态"""
        try:
            # Check if an active run exists, if not, try to get experiment
            if mlflow.active_run():
                return True
            mlflow.get_experiment_by_name(self.experiment_name) #This might create if not exists with some backends
            return True
        except Exception as e:
            logger.error(f"MLflow connection error: {str(e)}")
            return False

    def _handle_mlflow_error(self):
        # Ensure MLflow run is ended
        if self.mlflow_run:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error ending MLflow run: {str(e)}")

def main():
    # Create experiments
    experiments = [
        YOLOExperiment("multiclass_detect", "dataset/label_data_dataset"),
        YOLOExperiment("singleclass_detect", "dataset/single_class_dataset"),
        YOLOExperiment("overfit_experiment", "dataset/overfit_label_data_dataset")
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
                exp._handle_mlflow_error()
            except Exception as e:
                logger.warning(f"Error ending MLflow run: {str(e)}")

if __name__ == "__main__":
    # Add a check for MLflow run status before calling main operations
    # This is a simplified example; a robust solution might involve a global MLflow utility
    # For YOLOExperiment, mlflow is initialized in __init__
    
    # Example:
    # if not YOLOExperiment._check_mlflow_connection_static(): # A static method if needed before __init__
    #     logger.error("MLflow is not available. Exiting.")
    #     # sys.exit(1) # Or handle gracefully
    
    main() 