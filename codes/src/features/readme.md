产生训练用的数据流程
1. 运行 features.sh
    会在folder下产生多组小数据 如 feature-L0-Eset0-100sets-T500-C100-set32.h5
    /home/wehe/scratch/data/
2. 运行 merge_features.sh
    将子数据合成在一个大的文件中 feature-L0-Eset0-100sets-T500-C100.h5
    `nice /data/wehe/conda/envs/sbi/bin/python3 -u ./src/features/merge_features.py`