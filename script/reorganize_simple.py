#!/usr/bin/env python3
"""
简化的数据集重新划分脚本
将数据集按照 70% : 20% : 10% 的比例重新分配到 train/val/test
"""

import os
import shutil
import random
from pathlib import Path

def reorganize_dataset():
    """重新组织数据集"""
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 路径设置
    base_dir = Path.cwd().parent  # 从script目录到label_data_PRTest目录
    dataset_root = base_dir / "label_data_dataset"
    
    print(f"工作目录: {base_dir}")
    print(f"数据集目录: {dataset_root}")
    
    # 收集所有已经存在的图片和标签文件
    train_images = list((dataset_root / "images" / "train").glob("*.jpg"))
    val_images = list((dataset_root / "images" / "val").glob("*.jpg"))
    test_images = list((dataset_root / "test").glob("*.jpg"))
    
    train_labels = list((dataset_root / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_root / "labels" / "val").glob("*.txt"))
    
    print(f"当前分布:")
    print(f"训练集: {len(train_images)} 张图片, {len(train_labels)} 个标签")
    print(f"验证集: {len(val_images)} 张图片, {len(val_labels)} 个标签")
    print(f"测试集: {len(test_images)} 张图片")
    
    # 合并所有图片文件名（不含路径和扩展名）
    all_image_names = set()
    
    for img_list in [train_images, val_images, test_images]:
        for img_path in img_list:
            all_image_names.add(img_path.stem)
    
    # 只保留有对应标签文件的图片
    valid_names = []
    for name in all_image_names:
        # 检查是否有对应的标签文件
        label_found = False
        for label_path in train_labels + val_labels:
            if label_path.stem == name:
                label_found = True
                break
        
        if label_found:
            valid_names.append(name)
        else:
            print(f"警告: 图片 {name}.jpg 没有对应的标签文件，跳过")
    
    print(f"有效的图片-标签对: {len(valid_names)} 对")
    
    # 随机打乱
    random.shuffle(valid_names)
    
    # 计算分割点
    total = len(valid_names)
    train_count = int(total * 0.7)  # 70%
    val_count = int(total * 0.2)    # 20%
    test_count = total - train_count - val_count  # 剩余的给test
    
    print(f"\n新的分布:")
    print(f"训练集: {train_count} 张 ({train_count/total*100:.1f}%)")
    print(f"验证集: {val_count} 张 ({val_count/total*100:.1f}%)")
    print(f"测试集: {test_count} 张 ({test_count/total*100:.1f}%)")
    
    # 分割数据
    new_train_names = valid_names[:train_count]
    new_val_names = valid_names[train_count:train_count + val_count]
    new_test_names = valid_names[train_count + val_count:]
    
    # 创建新的目录结构
    for split in ['train', 'val', 'test']:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 清空现有目录
    for split in ['train', 'val', 'test']:
        for file_type in ['images', 'labels']:
            target_dir = dataset_root / file_type / split
            for file in target_dir.glob("*"):
                file.unlink()
    
    # 删除原来的test目录（如果存在）
    old_test_dir = dataset_root / "test"
    if old_test_dir.exists():
        shutil.rmtree(old_test_dir)
    
    def copy_files(names, split):
        """复制文件到指定的split目录"""
        for name in names:
            # 复制图片文件
            src_img = None
            for img_list in [train_images, val_images, test_images]:
                for img_path in img_list:
                    if img_path.stem == name:
                        src_img = img_path
                        break
                if src_img:
                    break
            
            if src_img:
                dst_img = dataset_root / "images" / split / f"{name}.jpg"
                shutil.copy2(src_img, dst_img)
            
            # 复制标签文件
            src_label = None
            for label_path in train_labels + val_labels:
                if label_path.stem == name:
                    src_label = label_path
                    break
            
            if src_label:
                dst_label = dataset_root / "labels" / split / f"{name}.txt"
                shutil.copy2(src_label, dst_label)
    
    # 执行文件复制
    print("\n开始重新组织文件...")
    copy_files(new_train_names, 'train')
    copy_files(new_val_names, 'val')
    copy_files(new_test_names, 'test')
    
    # 验证结果
    print("\n验证重组结果:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((dataset_root / "images" / split).glob("*.jpg")))
        label_count = len(list((dataset_root / "labels" / split).glob("*.txt")))
        print(f"{split}: {img_count} 张图片, {label_count} 个标签")
    
    print("\n数据集重组完成！")

if __name__ == "__main__":
    reorganize_dataset() 