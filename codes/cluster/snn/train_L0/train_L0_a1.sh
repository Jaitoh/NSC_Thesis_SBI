#!/bin/bash 
# generate dataset
python3 -u ./src/train/train_L0.py \
--run_simulator \
--config_simulator_path './src/config/simulator_Ca_Pb_Ma.yaml' \
--config_dataset_path './src/config/dataset_Sa0_suba1_Ra0.yaml' \
--config_train_path './src/config/train_Ta1.yaml' \
--log_dir './src/train/logs/log-simulator_Ca_Pb_Ma-dataset_Sa0_suba1_Ra0-train_Ta1' \
--gpu \
-y > ./cluster/snn/train_L0/train_L0_a1.log

echo 'finished simulation'