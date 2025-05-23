#!/usr/bin/env python3
"""
数据集重新划分脚本
将数据集按照 70% : 20% : 10% 的比例重新分配到 train/val/test
"""

import os
import shutil
import random
from pathlib import Path
import glob

def reorganize_dataset():
    """重新组织数据集"""
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 路径设置
    dataset_root = Path("../label_data_dataset")
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    
    # 创建临时目录来存储所有文件
    temp_dir = Path("../temp_reorganize")
    temp_dir.mkdir(exist_ok=True)
    
    # 收集所有图片文件
    all_images = []
    
    # 从train目录收集
    train_images = glob.glob(str(images_dir / "train" / "*.jpg"))
    train_labels = glob.glob(str(labels_dir / "train" / "*.txt"))
    
    # 从val目录收集
    val_images = glob.glob(str(images_dir / "val" / "*.jpg"))
    val_labels = glob.glob(str(labels_dir / "val" / "*.txt"))
    
    # 从test目录收集（如果有的话）
    test_images = glob.glob(str(dataset_root / "test" / "*.jpg"))
    
    # 合并所有图片
    all_images = train_images + val_images + test_images
    
    print(f"找到总计 {len(all_images)} 张图片")
    
    # 提取文件名（不含路径和扩展名）
    image_names = []
    for img_path in all_images:
        name = Path(img_path).stem
        # 验证是否有对应的标签文件
        label_exists = False
        for label_path in train_labels + val_labels:
            if Path(label_path).stem == name:
                label_exists = True
                break
        
        if label_exists:
            image_names.append(name)
        else:
            print(f"警告: 图片 {name}.jpg 没有对应的标签文件")
    
    print(f"有效的图片-标签对: {len(image_names)} 对")
    
    # 随机打乱
    random.shuffle(image_names)
    
    # 计算分割点
    total = len(image_names)
    train_count = int(total * 0.7)  # 70%
    val_count = int(total * 0.2)    # 20%
    test_count = total - train_count - val_count  # 剩余的给test
    
    print(f"\n新的分布:")
    print(f"训练集: {train_count} 张 ({train_count/total*100:.1f}%)")
    print(f"验证集: {val_count} 张 ({val_count/total*100:.1f}%)")
    print(f"测试集: {test_count} 张 ({test_count/total*100:.1f}%)")
    
    # 分割数据
    train_names = image_names[:train_count]
    val_names = image_names[train_count:train_count + val_count]
    test_names = image_names[train_count + val_count:]
    
    # 备份原始数据
    backup_dir = Path("../dataset_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(dataset_root, backup_dir)
    print(f"\n原始数据已备份到: {backup_dir}")
    
    # 清空现有目录
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            target_dir = dataset_root / subdir / split
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建test目录结构
    test_images_dir = dataset_root / "images" / "test"
    test_labels_dir = dataset_root / "labels" / "test"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动文件到新位置
    def move_files(names, split):
        for name in names:
            # 移动图片
            src_img = None
            for img_path in all_images:
                if Path(img_path).stem == name:
                    src_img = img_path
                    break
            
            if src_img:
                dst_img = dataset_root / "images" / split / f"{name}.jpg"
                shutil.copy2(src_img, dst_img)
            
            # 移动标签
            src_label = None
            for label_path in train_labels + val_labels:
                if Path(label_path).stem == name:
                    src_label = label_path
                    break
            
            if src_label:
                dst_label = dataset_root / "labels" / split / f"{name}.txt"
                shutil.copy2(src_label, dst_label)
    
    # 执行文件移动
    print("\n开始重新组织文件...")
    move_files(train_names, 'train')
    move_files(val_names, 'val')
    move_files(test_names, 'test')
    
    # 清理备份中的旧test目录（如果存在）
    old_test_dir = dataset_root / "test"
    if old_test_dir.exists():
        shutil.rmtree(old_test_dir)
    
    # 验证结果
    print("\n验证重组结果:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((dataset_root / "images" / split).glob("*.jpg")))
        label_count = len(list((dataset_root / "labels" / split).glob("*.txt")))
        print(f"{split}: {img_count} 张图片, {label_count} 个标签")
    
    print("\n数据集重组完成！")
    print("可以删除备份目录如果确认无误: rm -rf ../dataset_backup")

if __name__ == "__main__":
    reorganize_dataset() 