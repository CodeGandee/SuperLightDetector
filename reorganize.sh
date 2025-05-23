#!/bin/bash

echo "开始重新划分数据集..."

# 创建新的目录结构
mkdir -p label_data_dataset/images/{train,val,test}
mkdir -p label_data_dataset/labels/{train,val,test}

# 清空现有目录
rm -f label_data_dataset/images/train/*
rm -f label_data_dataset/images/val/*
rm -f label_data_dataset/images/test/*
rm -f label_data_dataset/labels/train/*
rm -f label_data_dataset/labels/val/*
rm -f label_data_dataset/labels/test/*

# 获取所有有对应标签的图片文件名
image_files=()
for img in dataset_backup/images/train/*.jpg dataset_backup/images/val/*.jpg; do
    if [ -f "$img" ]; then
        basename=$(basename "$img" .jpg)
        # 检查是否有对应的标签文件
        if [ -f "dataset_backup/labels/train/${basename}.txt" ] || [ -f "dataset_backup/labels/val/${basename}.txt" ]; then
            image_files+=("$basename")
        fi
    fi
done

echo "找到 ${#image_files[@]} 个有效的图片-标签对"

# 打乱数组
shuffled=($(printf '%s\n' "${image_files[@]}" | shuf))

# 计算分割点
total=${#shuffled[@]}
train_count=$((total * 7 / 10))  # 70%
val_count=$((total * 2 / 10))    # 20%
test_count=$((total - train_count - val_count))  # 剩余

echo "新的分布:"
echo "训练集: $train_count 张 ($(echo "scale=1; $train_count*100/$total" | bc)%)"
echo "验证集: $val_count 张 ($(echo "scale=1; $val_count*100/$total" | bc)%)"
echo "测试集: $test_count 张 ($(echo "scale=1; $test_count*100/$total" | bc)%)"

# 复制文件到新位置
copy_files() {
    local start=$1
    local end=$2
    local split=$3
    
    for ((i=start; i<end; i++)); do
        name=${shuffled[$i]}
        
        # 复制图片
        for src_img in dataset_backup/images/train/${name}.jpg dataset_backup/images/val/${name}.jpg; do
            if [ -f "$src_img" ]; then
                cp "$src_img" "label_data_dataset/images/${split}/"
                break
            fi
        done
        
        # 复制标签
        for src_label in dataset_backup/labels/train/${name}.txt dataset_backup/labels/val/${name}.txt; do
            if [ -f "$src_label" ]; then
                cp "$src_label" "label_data_dataset/labels/${split}/"
                break
            fi
        done
    done
}

echo "开始复制文件..."

# 训练集
copy_files 0 $train_count "train"

# 验证集
copy_files $train_count $((train_count + val_count)) "val"

# 测试集
copy_files $((train_count + val_count)) $total "test"

# 验证结果
echo "验证重组结果:"
for split in train val test; do
    img_count=$(ls label_data_dataset/images/$split/*.jpg 2>/dev/null | wc -l)
    label_count=$(ls label_data_dataset/labels/$split/*.txt 2>/dev/null | wc -l)
    echo "$split: $img_count 张图片, $label_count 个标签"
done

echo "数据集重组完成！" 