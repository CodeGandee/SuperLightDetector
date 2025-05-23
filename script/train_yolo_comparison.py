from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    log_dir = Path("label_data_PR/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('yolo_comparison')
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(
        log_dir / f"yolo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
output_dir = Path("label_data_PR")
output_dir.mkdir(exist_ok=True)

# 定义要测试的模型版本（按可用性排序）
model_versions = [
    "yolo12n",    # YOLO12n
    "yolo11n",  # YOLOv11n 
    "yolov10n",  # YOLOv10n
    "yolov9t",   # YOLOv9t
    "yolov8n",   # YOLOv8n (最稳定)
    "yolov5nu",  # YOLOv5n ultralytics版本
]

# 存储所有模型的结果
all_results = []

def log_and_print(message):
    """同时打印到控制台和记录到日志"""
    logger.info(message)
    print(message)

def check_model_availability(model_name):
    """检查模型是否可用"""
    try:
        model = YOLO(f"{model_name}.pt")
        return True
    except Exception as e:
        log_and_print(f"模型 {model_name} 不可用: {str(e)}")
        return False

def train_and_evaluate_model(model_name):
    """训练和评估单个模型"""
    log_and_print(f"\n{'='*60}")
    log_and_print(f"开始处理模型: {model_name}")
    log_and_print(f"{'='*60}")
    
    try:
        # 检查模型可用性
        if not check_model_availability(model_name):
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
                'status': 'unavailable',
                'error': 'Model not available'
            }
        
        # 加载模型
        model = YOLO(f"{model_name}.pt")
        log_and_print(f"✅ 成功加载模型: {model_name}.pt")
        
        # 训练模型
        log_and_print(f"🚀 开始训练 {model_name}...")
        train_results = model.train(
            data="label_data.yaml",
            epochs=200,  # 减少轮数以快速比较
            imgsz=640,
            device="2,3",
            batch=4,
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
            plots=True,  # 关闭训练图表以节省时间
            verbose=False,
            project="label_data_PR",
            name=f"{model_name}_comparison_200",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            close_mosaic=10,
            resume=False,
            amp=False,
            fraction=1.0,
            profile=False,
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
        
        log_and_print(f"✅ 模型 {model_name} 训练完成")
        
        # 验证集评估
        log_and_print(f"📊 开始验证集评估 {model_name}...")
        val_metrics = model.val(
            data="label_data.yaml",
            split='val',
            plots=False,  # 关闭验证图表
            save_json=False,
            verbose=False
        )
        
        # 测试集评估
        log_and_print(f"🔬 开始测试集评估 {model_name}...")
        test_metrics = model.val(
            data="label_data.yaml",
            split='test',
            plots=False,
            save_json=False,
            verbose=False
        )
        
        # 收集结果
        result = {
            'model': model_name,
            'val_mAP50': float(val_metrics.box.map50),
            'val_mAP50_95': float(val_metrics.box.map),
            'val_precision': float(val_metrics.box.mp),
            'val_recall': float(val_metrics.box.mr),
            'test_mAP50': float(test_metrics.box.map50),
            'test_mAP50_95': float(test_metrics.box.map),
            'test_precision': float(test_metrics.box.mp),
            'test_recall': float(test_metrics.box.mr),
            'status': 'success'
        }
        
        log_and_print(f"📈 模型 {model_name} 评估结果:")
        log_and_print(f"   验证集 - mAP50: {val_metrics.box.map50:.4f}, mAP50-95: {val_metrics.box.map:.4f}")
        log_and_print(f"   验证集 - 精确度: {val_metrics.box.mp:.4f}, 召回率: {val_metrics.box.mr:.4f}")
        log_and_print(f"   测试集 - mAP50: {test_metrics.box.map50:.4f}, mAP50-95: {test_metrics.box.map:.4f}")
        log_and_print(f"   测试集 - 精确度: {test_metrics.box.mp:.4f}, 召回率: {test_metrics.box.mr:.4f}")
        
        return result
        
    except Exception as e:
        log_and_print(f"❌ 模型 {model_name} 处理失败: {str(e)}")
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
    log_and_print(f"📄 结果已保存到CSV: {csv_path}")
    return df

def create_comparison_charts(df):
    """创建模型比较图表"""
    # 设置matplotlib字体
    try:
        # 尝试寻找中文字体
        font_paths = [
            'SimSun.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Arial.ttf',
            '/usr/share/fonts/TTF/SimSun.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        
        chinese_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                chinese_font = fm.FontProperties(fname=font_path)
                break
        
        if chinese_font is None:
            # 如果没有找到特定字体，使用系统默认字体
            chinese_font = fm.FontProperties(family='sans-serif')
            
    except Exception as e:
        log_and_print(f"⚠️ 字体设置失败，使用默认字体: {e}")
        chinese_font = fm.FontProperties(family='sans-serif')
    
    # 只显示成功的模型
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        log_and_print("⚠️  没有成功的模型结果，无法生成图表")
        return
    
    log_and_print(f"📊 生成 {len(df_success)} 个成功模型的比较图表...")
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('YOLO Model Performance Comparison', fontsize=18, fontweight='bold', y=0.95, fontproperties=chinese_font)
    
    models = df_success['model'].tolist()
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    # 1. mAP50比较
    x_pos = np.arange(len(models))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, df_success['val_mAP50'], width, 
                   label='Validation', color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0, 0].bar(x_pos + width/2, df_success['test_mAP50'], width, 
                   label='Test', color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('mAP@0.5 Performance Comparison', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    axes[0, 0].set_ylabel('mAP@0.5', fontsize=12, fontproperties=chinese_font)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=15, ha='right', fontproperties=chinese_font)
    axes[0, 0].legend(fontsize=11, prop=chinese_font)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].set_ylim(0, 1.1)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_mAP50'], df_success['test_mAP50'])):
        axes[0, 0].text(i-width/2, val_score + 0.02, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
        axes[0, 0].text(i+width/2, test_score + 0.02, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
    
    # 2. mAP50-95比较
    axes[0, 1].bar(x_pos - width/2, df_success['val_mAP50_95'], width, 
                   label='Validation', color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0, 1].bar(x_pos + width/2, df_success['test_mAP50_95'], width, 
                   label='Test', color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('mAP@0.5:0.95 Performance Comparison', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    axes[0, 1].set_ylabel('mAP@0.5:0.95', fontsize=12, fontproperties=chinese_font)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=15, ha='right', fontproperties=chinese_font)
    axes[0, 1].legend(fontsize=11, prop=chinese_font)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim(0, 1.0)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_mAP50_95'], df_success['test_mAP50_95'])):
        axes[0, 1].text(i-width/2, val_score + 0.02, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
        axes[0, 1].text(i+width/2, test_score + 0.02, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
    
    # 3. 精确度比较
    axes[1, 0].bar(x_pos - width/2, df_success['val_precision'], width, 
                   label='Validation', color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1, 0].bar(x_pos + width/2, df_success['test_precision'], width, 
                   label='Test', color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('Precision Performance Comparison', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    axes[1, 0].set_ylabel('Precision', fontsize=12, fontproperties=chinese_font)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=15, ha='right', fontproperties=chinese_font)
    axes[1, 0].legend(fontsize=11, prop=chinese_font)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_ylim(0, 1.1)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_precision'], df_success['test_precision'])):
        axes[1, 0].text(i-width/2, val_score + 0.02, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
        axes[1, 0].text(i+width/2, test_score + 0.02, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
    
    # 4. 召回率比较
    axes[1, 1].bar(x_pos - width/2, df_success['val_recall'], width, 
                   label='Validation', color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1, 1].bar(x_pos + width/2, df_success['test_recall'], width, 
                   label='Test', color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('Recall Performance Comparison', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    axes[1, 1].set_ylabel('Recall', fontsize=12, fontproperties=chinese_font)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=15, ha='right', fontproperties=chinese_font)
    axes[1, 1].legend(fontsize=11, prop=chinese_font)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].set_ylim(0, 1.1)
    
    # 添加数值标签
    for i, (val_score, test_score) in enumerate(zip(df_success['val_recall'], df_success['test_recall'])):
        axes[1, 1].text(i-width/2, val_score + 0.02, f'{val_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
        axes[1, 1].text(i+width/2, test_score + 0.02, f'{test_score:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', fontproperties=chinese_font)
    
    plt.tight_layout()
    chart_path = output_dir / "model_comparison_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    log_and_print(f"📊 比较图表已保存到: {chart_path}")
    
    # 创建雷达图
    create_radar_chart(df_success)

def create_radar_chart(df_success):
    """创建雷达图显示模型综合性能"""
    if len(df_success) == 0:
        return
    
    # 设置matplotlib字体
    try:
        # 尝试寻找中文字体
        font_paths = [
            'SimSun.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Arial.ttf',
            '/usr/share/fonts/TTF/SimSun.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        
        chinese_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                chinese_font = fm.FontProperties(fname=font_path)
                break
        
        if chinese_font is None:
            chinese_font = fm.FontProperties(family='sans-serif')
            
    except Exception as e:
        log_and_print(f"⚠️ 雷达图字体设置失败，使用默认字体: {e}")
        chinese_font = fm.FontProperties(family='sans-serif')
        
    log_and_print("🎯 生成模型性能雷达图...")
    
    # 准备雷达图数据 - 使用英文标签避免字体问题
    categories = ['Val mAP50', 'Val mAP50-95', 'Val Precision', 'Val Recall', 
                  'Test mAP50', 'Test mAP50-95', 'Test Precision', 'Test Recall']
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df_success)))
    
    for idx, (_, row) in enumerate(df_success.iterrows()):
        values = [
            row['val_mAP50'], row['val_mAP50_95'], row['val_precision'], row['val_recall'],
            row['test_mAP50'], row['test_mAP50_95'], row['test_precision'], row['test_recall']
        ]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontproperties=chinese_font)
    ax.set_ylim(0, 1)
    ax.set_title('YOLO Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20, fontproperties=chinese_font)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=11, prop=chinese_font)
    ax.grid(True)
    
    radar_path = output_dir / "model_performance_radar.png"
    plt.savefig(radar_path, dpi=300, bbox_inches='tight', facecolor='white')
    log_and_print(f"🎯 雷达图已保存到: {radar_path}")

def create_summary_table(df):
    """创建详细的汇总表格"""
    log_and_print("\n" + "="*100)
    log_and_print("🏆 YOLO模型性能详细汇总表")
    log_and_print("="*100)
    
    # 显示所有模型状态
    log_and_print("\n📋 模型状态总览:")
    log_and_print("-" * 50)
    for _, row in df.iterrows():
        status_emoji = "✅" if row['status'] == 'success' else "❌" if row['status'] == 'failed' else "⚠️"
        log_and_print(f"{status_emoji} {row['model']:<15} - {row['status']}")
    
    # 只显示成功的模型的详细结果
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        log_and_print("\n❌ 没有成功的模型结果")
        return
    
    log_and_print(f"\n📊 成功训练的模型详细性能表 ({len(df_success)}个模型):")
    log_and_print("-" * 120)
    log_and_print(f"{'模型':<12} {'验证mAP50':<11} {'验证mAP50-95':<13} {'验证精确度':<11} {'验证召回率':<11} "
                  f"{'测试mAP50':<11} {'测试mAP50-95':<13} {'测试精确度':<11} {'测试召回率':<11}")
    log_and_print("=" * 120)
    
    for _, row in df_success.iterrows():
        log_and_print(f"{row['model']:<12} {row['val_mAP50']:<11.4f} {row['val_mAP50_95']:<13.4f} "
                      f"{row['val_precision']:<11.4f} {row['val_recall']:<11.4f} "
                      f"{row['test_mAP50']:<11.4f} {row['test_mAP50_95']:<13.4f} "
                      f"{row['test_precision']:<11.4f} {row['test_recall']:<11.4f}")
    
    # 计算平均性能
    log_and_print("-" * 120)
    avg_row = df_success.select_dtypes(include=[np.number]).mean()
    log_and_print(f"{'平均值':<12} {avg_row['val_mAP50']:<11.4f} {avg_row['val_mAP50_95']:<13.4f} "
                  f"{avg_row['val_precision']:<11.4f} {avg_row['val_recall']:<11.4f} "
                  f"{avg_row['test_mAP50']:<11.4f} {avg_row['test_mAP50_95']:<13.4f} "
                  f"{avg_row['test_precision']:<11.4f} {avg_row['test_recall']:<11.4f}")
    
    # 找出最佳模型
    metrics = ['val_mAP50', 'test_mAP50', 'val_mAP50_95', 'test_mAP50_95']
    log_and_print("\n🏆 各项指标最佳模型:")
    log_and_print("=" * 50)
    
    for metric in metrics:
        best_idx = df_success[metric].idxmax()
        best_model = df_success.loc[best_idx]
        metric_name = metric.replace('val_', '验证集').replace('test_', '测试集').replace('mAP50_95', 'mAP50-95')
        log_and_print(f"🥇 {metric_name:<15}: {best_model['model']:<12} ({best_model[metric]:.4f})")

# 主程序
if __name__ == "__main__":
    log_and_print("🚀 开始YOLO模型全面比较实验")
    log_and_print(f"📋 计划测试的模型: {model_versions}")
    log_and_print(f"📊 数据集信息: 训练集35张, 验证集10张, 测试集6张")
    log_and_print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 训练和评估所有模型
    for i, model_name in enumerate(model_versions, 1):
        log_and_print(f"\n📅 进度: {i}/{len(model_versions)} - 当前模型: {model_name}")
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
    log_and_print(f"📄 详细结果已保存到JSON: {json_path}")
    
    # 最终总结
    successful_models = len([r for r in all_results if r['status'] == 'success'])
    log_and_print("\n" + "="*80)
    log_and_print("🎉 YOLO模型比较实验完成!")
    log_and_print(f"✅ 成功训练模型数: {successful_models}/{len(model_versions)}")
    log_and_print(f"📁 所有结果文件保存在: {output_dir}")
    log_and_print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print("="*80) 