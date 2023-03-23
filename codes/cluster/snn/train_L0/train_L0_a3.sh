#!/bin/bash 
# generate dataset
# --run_simulator \
python3 -u ./src/train/train_L0.py \
--config_simulator_path './src/config/simulator_Ca_Pb_Ma.yaml' \
--config_dataset_path './src/config/dataset_Sa0_suba0_Ra1.yaml' \
--config_train_path './src/config/train_Ta1.yaml' \
--log_dir './src/train/logs/log-simulator_Ca_Pb_Ma-dataset_Sa0_suba0_Ra1-train_Ta1' \
> ./cluster/snn/train_L0/train_L0_a3.log
# --gpu \

echo 'finished simulation'