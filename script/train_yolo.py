from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import sys
from datetime import datetime
import json

# 设置日志记录
def setup_logging():
    """设置日志记录到文件和控制台"""
    log_dir = Path("training_outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('yolo_training')
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(
        log_dir / f"yolo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# 创建输出目录
output_dir = Path("training_outputs")
output_dir.mkdir(exist_ok=True)

# 定义要测试的模型版本
model_versions = [
    "yolov8n",  # YOLOv8n (确保存在)
    "yolov5n",  # YOLOv5n
    "yolov9n",  # YOLOv9n (如果存在)
    "yolov10n", # YOLOv10n (如果存在)
    "yolov11n", # YOLOv11n (如果存在)
]

# 存储所有模型的结果
all_results = []

def log_and_print(message):
    """同时打印到控制台和记录到日志"""
    logger.info(message)
    print(message)

def train_and_evaluate_model(model_name):
    """训练和评估单个模型"""
    log_and_print(f"\n{'='*50}")
    log_and_print(f"开始训练模型: {model_name}")
    log_and_print(f"{'='*50}")
    
    try:
        # 加载模型
        model = YOLO(f"{model_name}.pt")
        log_and_print(f"成功加载模型: {model_name}.pt")
        
        # 训练模型
        log_and_print(f"开始训练 {model_name}...")
        train_results = model.train(
            data="label_data.yaml",
            epochs=100,  # 减少轮数以便快速比较多个模型
            imgsz=640,
            device="2,3",
            batch=4,
            lr0=0.001,
            lrf=0.1,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            save=True,
            plots=True,
            verbose=False,  # 减少输出
            project="training_outputs",
            name=f"{model_name}_comparison",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            close_mosaic=10,
            resume=False,
            amp=False,
            fraction=1.0,
            profile=False,
            freeze=None,
            # 数据增强参数
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
            copy_paste=0.0,
        )
        
        log_and_print(f"模型 {model_name} 训练完成")
        
        # 验证集评估
        log_and_print(f"开始验证集评估 {model_name}...")
        val_metrics = model.val(
            data="label_data.yaml",
            split='val',
            plots=True,
            save_json=True,
            verbose=False
        )
        
        # 测试集评估
        log_and_print(f"开始测试集评估 {model_name}...")
        test_metrics = model.val(
            data="label_data.yaml",
            split='test',
            plots=True,
            save_json=True,
            verbose=False
        )
        
        # 收集结果
        result = {
            'model': model_name,
            'val_mAP50': val_metrics.box.map50,
            'val_mAP50_95': val_metrics.box.map,
            'val_precision': val_metrics.box.mp,
            'val_recall': val_metrics.box.mr,
            'test_mAP50': test_metrics.box.map50,
            'test_mAP50_95': test_metrics.box.map,
            'test_precision': test_metrics.box.mp,
            'test_recall': test_metrics.box.mr,
            'status': 'success'
        }
        
        log_and_print(f"模型 {model_name} 结果:")
        log_and_print(f"  验证集 - mAP50: {val_metrics.box.map50:.4f}, mAP50-95: {val_metrics.box.map:.4f}")
        log_and_print(f"  验证集 - 精确度: {val_metrics.box.mp:.4f}, 召回率: {val_metrics.box.mr:.4f}")
        log_and_print(f"  测试集 - mAP50: {test_metrics.box.map50:.4f}, mAP50-95: {test_metrics.box.map:.4f}")
        log_and_print(f"  测试集 - 精确度: {test_metrics.box.mp:.4f}, 召回率: {test_metrics.box.mr:.4f}")
        
        return result
        
    except Exception as e:
        log_and_print(f"模型 {model_name} 训练失败: {str(e)}")
        return {
            'model': model_name,
            'val_mAP50': 0,
            'val_mAP50_95': 0,
            'val_precision': 0,
            'val_recall': 0,
            'test_mAP50': 0,
            'test_mAP50_95': 0,
            'test_precision': 0,
            'test_recall': 0,
            'status': 'failed',
            'error': str(e)
        }

def save_results_to_csv(results, filename="model_comparison_results.csv"):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)
    log_and_print(f"结果已保存到: {csv_path}")
    return df

def create_comparison_charts(df):
    """创建模型比较图表"""
    # 只显示成功的模型
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        log_and_print("没有成功的模型结果，无法生成图表")
        return
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('YOLO模型性能比较', fontsize=16, fontweight='bold')
    
    models = df_success['model'].tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # 1. mAP50比较
    x_pos = np.arange(len(models))
    axes[0, 0].bar(x_pos - 0.2, df_success['val_mAP50'], 0.4, 
                   label='验证集', color=colors, alpha=0.7)
    axes[0, 0].bar(x_pos + 0.2, df_success['test_mAP50'], 0.4, 
                   label='测试集', color=colors, alpha=0.9)
    axes[0, 0].set_title('mAP@0.5 比较')
    axes[0, 0].set_ylabel('mAP@0.5')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_mAP50'], df_success['test_mAP50'])):
        axes[0, 0].text(i-0.2, val_score + 0.01, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i+0.2, test_score + 0.01, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 2. mAP50-95比较
    axes[0, 1].bar(x_pos - 0.2, df_success['val_mAP50_95'], 0.4, 
                   label='验证集', color=colors, alpha=0.7)
    axes[0, 1].bar(x_pos + 0.2, df_success['test_mAP50_95'], 0.4, 
                   label='测试集', color=colors, alpha=0.9)
    axes[0, 1].set_title('mAP@0.5:0.95 比较')
    axes[0, 1].set_ylabel('mAP@0.5:0.95')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_mAP50_95'], df_success['test_mAP50_95'])):
        axes[0, 1].text(i-0.2, val_score + 0.01, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        axes[0, 1].text(i+0.2, test_score + 0.01, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 3. 精确度比较
    axes[1, 0].bar(x_pos - 0.2, df_success['val_precision'], 0.4, 
                   label='验证集', color=colors, alpha=0.7)
    axes[1, 0].bar(x_pos + 0.2, df_success['test_precision'], 0.4, 
                   label='测试集', color=colors, alpha=0.9)
    axes[1, 0].set_title('精确度比较')
    axes[1, 0].set_ylabel('精确度')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_precision'], df_success['test_precision'])):
        axes[1, 0].text(i-0.2, val_score + 0.01, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        axes[1, 0].text(i+0.2, test_score + 0.01, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 4. 召回率比较
    axes[1, 1].bar(x_pos - 0.2, df_success['val_recall'], 0.4, 
                   label='验证集', color=colors, alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, df_success['test_recall'], 0.4, 
                   label='测试集', color=colors, alpha=0.9)
    axes[1, 1].set_title('召回率比较')
    axes[1, 1].set_ylabel('召回率')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_recall'], df_success['test_recall'])):
        axes[1, 1].text(i-0.2, val_score + 0.01, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(i+0.2, test_score + 0.01, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    chart_path = output_dir / "model_comparison_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    log_and_print(f"比较图表已保存到: {chart_path}")

def create_summary_table(df):
    """创建汇总表格"""
    log_and_print("\n" + "="*80)
    log_and_print("模型性能汇总表")
    log_and_print("="*80)
    
    # 只显示成功的模型
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        log_and_print("没有成功的模型结果")
        return
    
    # 创建格式化的表格
    headers = ["模型", "验证mAP50", "验证mAP50-95", "验证精确度", "验证召回率", 
               "测试mAP50", "测试mAP50-95", "测试精确度", "测试召回率"]
    
    log_and_print(f"{'模型':<12} {'验证mAP50':<10} {'验证mAP50-95':<12} {'验证精确度':<10} {'验证召回率':<10} "
                  f"{'测试mAP50':<10} {'测试mAP50-95':<12} {'测试精确度':<10} {'测试召回率':<10}")
    log_and_print("-" * 120)
    
    for _, row in df_success.iterrows():
        log_and_print(f"{row['model']:<12} {row['val_mAP50']:<10.4f} {row['val_mAP50_95']:<12.4f} "
                      f"{row['val_precision']:<10.4f} {row['val_recall']:<10.4f} "
                      f"{row['test_mAP50']:<10.4f} {row['test_mAP50_95']:<12.4f} "
                      f"{row['test_precision']:<10.4f} {row['test_recall']:<10.4f}")
    
    # 找出最佳模型
    best_val_map50 = df_success.loc[df_success['val_mAP50'].idxmax()]
    best_test_map50 = df_success.loc[df_success['test_mAP50'].idxmax()]
    best_val_map50_95 = df_success.loc[df_success['val_mAP50_95'].idxmax()]
    best_test_map50_95 = df_success.loc[df_success['test_mAP50_95'].idxmax()]
    
    log_and_print("\n" + "="*40)
    log_and_print("最佳性能模型:")
    log_and_print("="*40)
    log_and_print(f"验证集mAP50最佳: {best_val_map50['model']} ({best_val_map50['val_mAP50']:.4f})")
    log_and_print(f"测试集mAP50最佳: {best_test_map50['model']} ({best_test_map50['test_mAP50']:.4f})")
    log_and_print(f"验证集mAP50-95最佳: {best_val_map50_95['model']} ({best_val_map50_95['val_mAP50_95']:.4f})")
    log_and_print(f"测试集mAP50-95最佳: {best_test_map50_95['model']} ({best_test_map50_95['test_mAP50_95']:.4f})")

# 主程序
if __name__ == "__main__":
    log_and_print("开始YOLO模型比较实验")
    log_and_print(f"计划测试的模型: {model_versions}")
    log_and_print(f"数据集信息: 训练集35张, 验证集10张, 测试集6张")
    
    # 训练和评估所有模型
    for model_name in model_versions:
        result = train_and_evaluate_model(model_name)
        all_results.append(result)
    
    # 保存结果到CSV
    df_results = save_results_to_csv(all_results)
    
    # 创建比较图表
    create_comparison_charts(df_results)
    
    # 创建汇总表格
    create_summary_table(df_results)
    
    # 保存详细的JSON结果
    json_path = output_dir / "detailed_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log_and_print(f"详细结果已保存到: {json_path}")
    
    log_and_print("\n" + "="*50)
    log_and_print("所有模型比较实验完成!")
    log_and_print(f"结果文件保存在: {output_dir}")
    log_and_print("="*50)


