from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# 设置简单的日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/quick_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 创建输出目录
output_dir = Path("quick_test_results")
output_dir.mkdir(exist_ok=True)

# 要测试的模型（只选择确定可用的）
models_to_test = [
    "YOLOv7n", 
    "yolov6n",
    "yolov4n",
    "yolov3n",
    "yolov2n",
    "yolov1n"

    
]

def quick_test_model(model_name):
    """快速测试单个模型"""
    logger.info(f"🚀 开始快速测试模型: {model_name}")
    
    try:
        # 加载并快速训练模型
        model = YOLO(f"./models/{model_name}.pt")
        logger.info(f"✅ 模型 {model_name} 加载成功")
        
        # 快速训练（很少的轮数）
        train_results = model.train(
            data="label_data_dataset/dataset.yaml",
            epochs=10,  # 只训练10轮进行快速测试
            imgsz=640,
            device="6,7",
            batch=4,
            verbose=False,
            plots=False,
            project="quick_test_results",
            name=f"{model_name}_quick",
            exist_ok=True,
        )
        
        # 验证
        val_metrics = model.val(data="label_data.yaml", split='val', verbose=False)
        test_metrics = model.val(data="label_data.yaml", split='test', verbose=False)
        
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
        
        logger.info(f"📊 {model_name} 结果 - 验证mAP50: {val_metrics.box.map50:.4f}, 测试mAP50: {test_metrics.box.map50:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 模型 {model_name} 测试失败: {e}")
        return {
            'model': model_name,
            'val_mAP50': 0, 'val_mAP50_95': 0, 'val_precision': 0, 'val_recall': 0,
            'test_mAP50': 0, 'test_mAP50_95': 0, 'test_precision': 0, 'test_recall': 0,
            'status': 'failed', 'error': str(e)
        }

def create_quick_chart(results_df):
    """创建快速比较图表"""
    df_success = results_df[results_df['status'] == 'success']
    if len(df_success) == 0:
        logger.warning("没有成功的结果用于绘图")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO模型快速性能比较', fontsize=16)
    
    models = df_success['model'].tolist()
    x = range(len(models))
    width = 0.35
    
    # mAP50
    ax1.bar([i-width/2 for i in x], df_success['val_mAP50'], width, label='验证集', alpha=0.8)
    ax1.bar([i+width/2 for i in x], df_success['test_mAP50'], width, label='测试集', alpha=0.8)
    ax1.set_title('mAP@0.5')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mAP50-95
    ax2.bar([i-width/2 for i in x], df_success['val_mAP50_95'], width, label='验证集', alpha=0.8)
    ax2.bar([i+width/2 for i in x], df_success['test_mAP50_95'], width, label='测试集', alpha=0.8)
    ax2.set_title('mAP@0.5:0.95')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 精确度
    ax3.bar([i-width/2 for i in x], df_success['val_precision'], width, label='验证集', alpha=0.8)
    ax3.bar([i+width/2 for i in x], df_success['test_precision'], width, label='测试集', alpha=0.8)
    ax3.set_title('精确度')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 召回率
    ax4.bar([i-width/2 for i in x], df_success['val_recall'], width, label='验证集', alpha=0.8)
    ax4.bar([i+width/2 for i in x], df_success['test_recall'], width, label='测试集', alpha=0.8)
    ax4.set_title('召回率')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / "quick_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 图表已保存到: {chart_path}")

if __name__ == "__main__":
    logger.info(f"🏁 开始快速模型测试 - {datetime.now()}")
    logger.info(f"测试模型: {models_to_test}")
    
    all_results = []
    
    # 测试每个模型
    for model_name in models_to_test:
        result = quick_test_model(model_name)
        all_results.append(result)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "quick_test_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"📄 结果已保存到: {csv_path}")
    
    # 打印结果表格
    logger.info("\n" + "="*80)
    logger.info("快速测试结果汇总:")
    logger.info("="*80)
    for _, row in df.iterrows():
        if row['status'] == 'success':
            logger.info(f"{row['model']:<10} - 验证mAP50: {row['val_mAP50']:.4f}, 测试mAP50: {row['test_mAP50']:.4f}")
        else:
            logger.info(f"{row['model']:<10} - 失败: {row.get('error', 'Unknown error')}")
    
    # 创建图表
    create_quick_chart(df)
    
    logger.info(f"🎉 快速测试完成 - {datetime.now()}")
    logger.info(f"所有结果保存在: {output_dir}")
    
    # 显示成功的模型数量
    success_count = len(df[df['status'] == 'success'])
    logger.info(f"✅ 成功测试模型数: {success_count}/{len(models_to_test)}") 