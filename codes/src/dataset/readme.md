产生训练用的数据流程
1. 运行 simulate_and_save.sh
    会在folder下产生多组小数据 如 dataset_part_0.h5
    /home/wehe/scratch/data/dataset/
2. 运行 merge_dataset.sh # TODO 添加自定义数据文件信息输入
    将想要的数据部分合成在一个大的文件中
3. 运行 process_dataset_x.sh # TODO 检查已处理文件/set，不重复处理
    会将 merge后的数据， 进行预处理 产生seqC-> norm/summary0/summary1