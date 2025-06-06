# 准备数据单元

# 形成训练集
python script/translate_coco_to_utralytics_yolo.py --coco_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604/dataset_train --output_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics --debug --split_ratio 1 0 0

# 追加测试集

python script/translate_coco_to_utralytics_yolo.py --coco_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604/dataset_test --output_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics --debug --split_ratio 0 0 1 --append

# 追加验证集
python script/translate_coco_to_utralytics_yolo.py --coco_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604/dataset_val --output_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics --debug --split_ratio 0 1 0 --append


# 形成单类别训练集
python script/create_single_class_dataset.py  --input_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics

# 从ultralytics转换为yolo格式
python script/convert_utltralytics_to_yolo.py --input_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics

# 从ultralytics转换为yolo格式
python script/convert_utltralytics_to_yolo.py --input_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/single_class_dataset/

# 从yolo转换为coco格式
python script/yolo2coco.py --input_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics_yolo --debug

# 从yolo转换为coco格式
python script/yolo2coco.py --input_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/single_class_dataset_yolo --debug

# 从yolo转换为coco格式(训练集、验证集、测试集)
python script/split_coco_from_yolo_lists.py --main_coco_json /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics_yolo_merge_coco/annotations.json --yolo_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics_yolo  --debug


# 从yolo转换为coco格式(训练集、验证集、测试集)
python script/split_coco_from_yolo_lists.py --main_coco_json /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/single_class_dataset_yolo_merge_coco/annotations.json --yolo_dir /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/single_class_dataset_yolo  --debug


cp -r /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/train_equal_test

cp -r /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics_yolo /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/train_equal_test_yolo


cp -r /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics_yolo_coco /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/train_equal_test_yolo_coco

# 开始模型训练

python script/train_yolo_experiments.py --config_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/config/config.yaml


bash ./models/nanodet/train_and_test.sh


bash ./models/picodet/tools/train_and_test.sh


python script/draw_png.py --base_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/experiments_white_dataset_0604_2/other_multiclass_detect --dataset_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics


python script/draw_png.py --base_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/experiments_white_dataset_0604_2/other_sigleclass_detect --dataset_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics


python script/draw_png.py --base_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/experiments_white_dataset_0604_2/other_overfit_experiment --dataset_path /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/dataset/white_dataset_0604_2/annotations_ultralytics
