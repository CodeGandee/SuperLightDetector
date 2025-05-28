# YOLO系列模型比较实验项目

本项目主要用于进行YOLO系列模型的比较实验，以及NanoDet和PicoDet等轻量级检测模型的性能对比实验。项目包含完整的实验脚本、数据集和模型权重文件。

## 项目结构

```
label_data_PRTest/
├── dataset/              # 存放实验数据集
├── models/              # 存放YOLO权重和其他检测模型
├── script/              # 实验相关脚本
│   ├── train_yolo_experiments.py    # YOLO系列模型训练和比较实验脚本
│   ├── draw_png.py                  # 实验结果可视化脚本
│   └── split_coco_from_yolo_lists.py # YOLO格式数据集转换为COCO格式工具
├── experiments/         # 实验输出结果
│   ├── multiclass_detect/          # 多类别检测实验
│   ├── singleclass_detect/         # 单类别检测实验
│   ├── overfit_experiment/         # 过拟合实验
│   ├── other_multiclass_detect/    # 其他模型多类别检测实验
│   ├── other_sigleclass_detect/    # 其他模型单类别检测实验
│   └── other_overfit_experiment/   # 其他模型过拟合实验
├── logs/               # 训练日志
└── requirements.txt    # 项目依赖
```

## 实验结果

### 1. 多类别检测实验 (multiclass_detect)
<div align="center">
<img src="experiments/multiclass_detect/mAP50_heatmap.png" alt="多类别检测mAP50热力图" width="800"/>
</div>

- 实验目的：评估不同模型在多类别目标检测任务上的性能
- 实验内容：使用YOLO系列模型和轻量级模型（NanoDet、PicoDet）进行多类别检测
- 结果说明：热力图展示了不同模型在不同训练轮次下的mAP50性能变化

### 2. 单类别检测实验 (singleclass_detect)
<div align="center">
<img src="experiments/singleclass_detect/mAP50_heatmap.png" alt="单类别检测mAP50热力图" width="800"/>
</div>

- 实验目的：评估模型在单类别检测任务上的性能表现
- 实验内容：使用相同模型架构进行单类别目标检测
- 结果说明：展示了模型在简化任务下的性能表现和收敛特性

### 3. 过拟合实验 (overfit_experiment)
<div align="center">
<img src="experiments/overfit_experiment/mAP50_heatmap.png" alt="过拟合实验mAP50热力图" width="800"/>
</div>

- 实验目的：研究模型在小数据集上的过拟合现象
- 实验内容：使用不同训练策略和模型配置进行过拟合实验
- 结果说明：展示了模型在训练过程中的性能变化和过拟合趋势

### 4. 其他模型多类别检测 (other_multiclass_detect)
<div align="center">
<img src="experiments/other_multiclass_detect/mAP50_heatmap.png" alt="其他模型多类别检测mAP50热力图" width="800"/>
</div>

- 实验目的：评估非YOLO系列模型在多类别检测任务上的性能
- 实验内容：使用NanoDet、PicoDet等轻量级模型进行多类别检测
- 结果说明：对比展示了轻量级模型与YOLO系列模型的性能差异

### 5. 其他模型单类别检测 (other_sigleclass_detect)
<div align="center">
<img src="experiments/other_sigleclass_detect/mAP50_heatmap.png" alt="其他模型单类别检测mAP50热力图" width="800"/>
</div>

- 实验目的：评估轻量级模型在单类别检测任务上的性能
- 实验内容：使用NanoDet、PicoDet等模型进行单类别检测
- 结果说明：展示了轻量级模型在简化任务下的性能表现

### 6. 其他模型过拟合实验 (other_overfit_experiment)
<div align="center">
<img src="experiments/other_overfit_experiment/mAP50_heatmap.png" alt="其他模型过拟合实验mAP50热力图" width="800"/>
</div>

- 实验目的：研究轻量级模型在小数据集上的过拟合特性
- 实验内容：使用不同轻量级模型配置进行过拟合实验
- 结果说明：展示了轻量级模型在训练过程中的性能变化和过拟合趋势

## 主要功能

### 1. 模型比较实验
- YOLO系列模型（YOLOv5、YOLOv8等）的性能对比
- NanoDet和PicoDet等轻量级检测模型的性能评估
- 不同模型在相同数据集上的训练效果对比

### 2. 实验脚本说明

#### train_yolo_experiments.py
- 用于执行YOLO系列模型的训练实验
- 支持多模型并行训练和比较
- 自动记录训练指标和性能数据
- 生成详细的实验报告

#### draw_png.py
- 用于可视化实验结果
- 生成模型性能对比图表
- 支持mAP、训练速度、推理速度等指标的对比展示
- 生成热力图形式的性能对比图

#### split_coco_from_yolo_lists.py
- 将YOLO格式的数据集转换为COCO格式
- 支持数据集格式转换和验证
- 生成COCO格式的标注文件

## 环境设置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行模型比较实验
```bash
cd script
python train_yolo_experiments.py
```

### 2. 生成实验结果可视化
```bash
cd script
python draw_png.py
```

### 3. 数据集格式转换
```bash
cd script
python split_coco_from_yolo_lists.py
```

## 实验输出

实验完成后，以下内容将保存在相应目录中：

### 1. 实验结果
- 模型性能指标对比
- 训练速度对比
- 推理速度对比
- 模型参数量对比

### 2. 可视化结果
- 性能对比图表
- 热力图形式的性能展示
- 训练过程曲线

### 3. 模型权重
- 各模型的最佳权重文件
- 训练过程中的检查点

## 注意事项

1. **数据集准备**: 确保数据集已正确放置在dataset目录下
2. **模型权重**: 预训练权重文件应放置在models目录下
3. **实验配置**: 可在脚本中修改实验参数和配置
4. **资源需求**: 根据实验规模准备足够的计算资源
5. **结果保存**: 定期备份实验结果和模型权重

## 性能评估

实验完成后，可以查看：
- 各模型的mAP指标对比
- 训练和推理速度对比
- 模型参数量和计算量对比
- 不同场景下的性能表现

## 故障排除

如果遇到问题：
1. 检查数据集路径和格式是否正确
2. 确认所有依赖已正确安装
3. 验证模型权重文件是否完整
4. 检查实验配置参数是否合理
5. 查看日志文件了解详细错误信息 