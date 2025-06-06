import os
import json
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from loguru import logger
import time
import argparse
from collections import defaultdict
from prettytable import PrettyTable
import yaml
# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "logs/translate_coco_to_utralytics_yolo_{time}.log",
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

chinese_to_english_dict_order = {
    "停车": "parking",
    "左转": "left_turn",
    "掉头": "turn_around",
    "限速": "speed_limit",
    "打架": "fight",
    "火警": "fire_alarm",
    "右转": "right_turn",
    "垃圾": "trash"
}

def coco2ultralytics(coco_dir, output_dir, split_ratio=[1,0,0],debug=False,append=False,sample_num=5):
    # 1. 路径准备
    coco_dir = Path(coco_dir)
    coco_json = coco_dir / 'annotations' / 'coco.json'
    coco_img_dir = coco_dir / 'images'
    output_dir = Path(output_dir)
    if not append and output_dir.exists() and any(output_dir.iterdir()):
        shutil.rmtree(output_dir)
        logger.info(f"output_dir {output_dir} exists and is not empty, removing...")
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    debug_dir = output_dir / 'debug'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preparing coco json starting...")
    time_begin = time.perf_counter_ns()
    # 2. 读取COCO
    with open(str(coco_json), 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']

    # 3. 类别映射（与COCO categories顺序严格一致）
    # 创建一个字典来存储类别名称到索引的映射
    category_order = {name: idx for idx, name in enumerate(chinese_to_english_dict_order.keys())}
    
    # 对categories进行排序，使用category_order中的顺序
    coco_cat_sorted = sorted(coco['categories'], 
                           key=lambda x: category_order.get(x['name'], float('inf')))
    
    # 处理中文到英文的转换
    for cat in coco_cat_sorted:
        chinese_name = cat['name']
        if chinese_name in chinese_to_english_dict_order:
            cat['name'] = chinese_to_english_dict_order[chinese_name]
            logger.info(f"Chinese category '{chinese_name}' translated to '{cat['name']}'")
        else:
            logger.error(f"Chinese category '{chinese_name}' not found in translation dictionary")
    id2name = {cat['id']: cat['name'] for cat in coco_cat_sorted}
    # names: id->name，顺序与coco categories一致
    names = {i: cat['name'] for i, cat in enumerate(coco_cat_sorted)}
    # 构建coco category id到names id的映射
    catid2namesid = {cat['id']: i for i, cat in enumerate(coco_cat_sorted)}

    # 4. 收集每张图片的标注
    imgid2annos = {}
    for anno in annotations:
        imgid2annos.setdefault(anno['image_id'], []).append(anno)
    time_cost = (time.perf_counter_ns() - time_begin) / 1e6
    logger.info(f"preparing coco json finished, time cost: {time_cost:.2f}ms")

    # 统计标签数量分布
    label_count_stats = defaultdict(int)
    max_labels = 0
    for img_id, annos in imgid2annos.items():
        label_count = len(annos)
        label_count_stats[label_count] += 1
        max_labels = max(max_labels, label_count)

    # 打印标签数量统计
    label_table = PrettyTable()
    label_table.field_names = ["Number of Labels", "Number of Images", "Percentage"]
    total_images_with_labels = sum(label_count_stats.values())
    
    for label_count in range(1, max_labels + 1):
        num_images = label_count_stats[label_count]
        percentage = (num_images / total_images_with_labels) * 100 if total_images_with_labels > 0 else 0
        label_table.add_row([label_count, num_images, f"{percentage:.2f}%"])
    
    logger.info(f"\nLabel Count Statistics:\n{label_table}")
    logger.info(f"Maximum number of labels in a single image: {max_labels}")

    # 5. 转换
    logger.info(f"Converting COCO to Ultralytics YOLO, {len(images)} images found, starting...")
    time_begin = time.perf_counter_ns()
    # 统计每个类型的图像数和标注数
    stats = defaultdict(lambda: {'images': 0, 'annotations': 0})
    for cat in coco_cat_sorted:
        stats[cat['name']] = {'images': 0, 'annotations': 0}

    # 创建训练、验证、测试目录
    train_img_dir = output_img_dir / 'train'
    val_img_dir = output_img_dir / 'val'
    test_img_dir = output_img_dir / 'test'
    train_label_dir = output_label_dir / 'train'
    val_label_dir = output_label_dir / 'val'
    test_label_dir = output_label_dir / 'test'
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir.mkdir(parents=True, exist_ok=True)

    # 处理数据集分割
    img_ids = list(images.keys())
    random.shuffle(img_ids)
    n = len(img_ids)
    
    # 检查是否只有一个比例为1，其他为0的情况
    if sum(1 for x in split_ratio if x == 1) == 1:
        # 找到比例为1的索引
        target_idx = split_ratio.index(1)
        if target_idx == 0:  # train
            train_ids = img_ids
            val_ids = []
            test_ids = []
        elif target_idx == 1:  # val
            train_ids = []
            val_ids = img_ids
            test_ids = []
        else:  # test
            train_ids = []
            val_ids = []
            test_ids = img_ids
    else:
        # 按比例分割数据集
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        train_ids = img_ids[:n_train]
        val_ids = img_ids[n_train:n_train + n_val]
        test_ids = img_ids[n_train + n_val:]

    # 转换并统计
    for img_id, img_info in tqdm(images.items(), desc="Converting COCO to Ultralytics YOLO", colour="green"):
        img_name = img_info['file_name']
        img_path = coco_img_dir / img_name
        
        # 确定输出目录
        if img_id in train_ids:
            out_img_dir = train_img_dir
            out_label_dir = train_label_dir
        elif img_id in val_ids:
            out_img_dir = val_img_dir
            out_label_dir = val_label_dir
        else:
            out_img_dir = test_img_dir
            out_label_dir = test_label_dir
        out_img_path = out_img_dir / img_name
        out_label_path = out_label_dir / f"{Path(img_name).stem}.txt"
        # 拷贝图片
        shutil.copy(str(img_path), str(out_img_path))
        # 获取图片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to read image: {img_path}")
            continue
        h, w = img.shape[:2]
        # 写标签
        lines = []
        for anno in imgid2annos.get(img_id, []):
            cat_id = anno['category_id']
            if cat_id not in catid2namesid:
                logger.error(f"Category ID {cat_id} not found in catid2namesid for {img_name}")
                continue
            class_id = catid2namesid[cat_id]  # 保证与names顺序一致
            cat_name = id2name[cat_id]
            stats[cat_name]['annotations'] += 1
            x, y, bw, bh = anno['bbox']
            # COCO: x,y为左上角，Ultralytics: x_center, y_center, w, h (归一化)
            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
        with open(out_label_path, 'w') as f:
            f.write('\n'.join(lines))
        # 统计图像数
        for cat_id in set(anno['category_id'] for anno in imgid2annos.get(img_id, [])):
            if cat_id in id2name:
                stats[id2name[cat_id]]['images'] += 1
    time_cost = (time.perf_counter_ns() - time_begin) / 1e6
    logger.info(f"Converting COCO to Ultralytics YOLO finished, time cost: {time_cost:.2f}ms")
    # 使用prettytable打印统计结果
    table = PrettyTable()
    table.field_names = ["Category", "Images", "Annotations", "Image Ratio", "Annotation Ratio"]
    total_images = sum(data['images'] for data in stats.values())
    total_annotations = sum(data['annotations'] for data in stats.values())
    
    # 统计每个类别的图像数（不重复计算）
    unique_images_per_category = defaultdict(set)
    for img_id, annos in imgid2annos.items():
        for anno in annos:
            cat_id = anno['category_id']
            if cat_id in id2name:
                unique_images_per_category[id2name[cat_id]].add(img_id)
    
    # 重新计算每个类别的图像数
    for cat_name, data in stats.items():
        img_ratio = len(unique_images_per_category[cat_name]) / len(images) if len(images) > 0 else 0
        anno_ratio = data['annotations'] / total_annotations if total_annotations > 0 else 0
        table.add_row([
            cat_name,
            len(unique_images_per_category[cat_name]),
            data['annotations'],
            f"{img_ratio:.2%}",
            f"{anno_ratio:.2%}"
        ])
    
    # 添加总计行
    total_image_table= sum(len(unique_images_per_category[cat_name]) for cat_name in stats.keys())
    total_img_ratio = sum(len(unique_images_per_category[cat_name]) for cat_name in stats.keys()) / len(images) if len(images) > 0 else 0
    total_anno_ratio = total_annotations / total_annotations if total_annotations > 0 else 0
    table.add_row([
        "Total",
        total_image_table ,
        total_annotations,
        f"{total_img_ratio:.2%}",
        f"{total_anno_ratio:.2%}"
    ])
    
    # 统计没有任何Annotations的图像数量
    img_with_anno = set()
    for img_id, annos in imgid2annos.items():
        if len(annos) > 0:
            img_with_anno.add(img_id)
    no_anno_images = len(images) - len(img_with_anno)
    no_anno_ratio = no_anno_images / len(images) if len(images) > 0 else 0
    logger.info(f"Dataset statistics:\n{table}")
    logger.info(f"Total Images in COCO JSON: {len(coco['images'])}")
    logger.info(f"Total Images with Annotations: {len(img_with_anno)}")
    logger.info(f"Total Annotations: {total_annotations}")
    logger.info(f"Images with NO Annotations: {no_anno_images} ({no_anno_ratio:.2%})")
    # 6. debug可视化
    if debug:
        logger.info(f"Generating debug images for {sample_num} images, starting...")
        time_begin = time.perf_counter_ns()
        sample_imgs = random.sample(list(images.values()), min(sample_num, len(images)))
        for img_info in sample_imgs:
            img_name = img_info['file_name']
            img_path = coco_img_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            h, w = img.shape[:2]
            for anno in imgid2annos.get(img_info['id'], []):
                cat_id = anno['category_id']
                if cat_id not in catid2namesid:
                    logger.error(f"Category ID {cat_id} not found in catid2namesid in debug mode for {img_name}")
                    continue
                class_id = catid2namesid[cat_id]
                cat_name = id2name[cat_id]
                x, y, bw, bh = anno['bbox']
                pt1 = (int(x), int(y))
                pt2 = (int(x + bw), int(y + bh))
                cv2.rectangle(img, pt1, pt2, (0,255,0), 2)
                # 使用中文显示类别名称
                logger.info(f"draw cat_name: {cat_name}  for {img_name}")
                cv2.putText(img, cat_name, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # 确保debug目录存在
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / img_name), img)
        time_cost = (time.perf_counter_ns() - time_begin) / 1e6
        logger.info(f"Generating debug images for {sample_num} images finished, time cost: {time_cost:.2f}ms")

    # 7. 自动生成dataset.yaml
    logger.info(f"Generating dataset.yaml starting...")
    time_begin = time.perf_counter_ns()
    yaml_path = output_dir / 'dataset.yaml'
    if not append or not yaml_path.exists():
        yaml_content = {
            'path': output_dir.as_posix(),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': names
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True)
    time_cost = (time.perf_counter_ns() - time_begin) / 1e6
    logger.info(f"Generating dataset.yaml finished, time cost: {time_cost:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", required=True,help="COCO格式数据集文件夹")
    parser.add_argument("--output_dir", required=True,help="Ultralytics格式输出文件夹")
    parser.add_argument("--split_ratio",nargs=3,type=float,default=[1,0,0],help="训练集、验证集、测试集比例")
    parser.add_argument("--debug", action="store_true", help="是否生成debug可视化")
    parser.add_argument("--append",action="store_true",help="是否要附加")
    args = parser.parse_args()
    coco2ultralytics(args.coco_dir, args.output_dir,args.split_ratio,args.debug,args.append) 
