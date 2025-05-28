# 切换到picodet的develop分支
cd /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/models/picodet && git checkout develop && cd ../nanodet && git checkout main

# 生成nanodet的patch
cd /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/models/nanodet && git diff origin/main > ../../models_patch/nanodet_custom.patch

# 生成picodet的patch
cd /nfs/3D/zhangleichao/zhangleichao/CLIMB_WS/label_data_PRTest/models/picodet && git diff origin/develop > ../../models_patch/picodet_custom.patch