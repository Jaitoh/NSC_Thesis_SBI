python3 -u ./src/train/train_L0.py \
--seed 0 \
--run_simulator \
--config_simulator_path './src/config/test_simulator.yaml' \
--config_dataset_path './src/config/test_dataset.yaml' \
--config_train_path './src/config/test_train.yaml' \
--log_dir './src/train/logs/log-test' \
--gpu \
-y \
> ./cluster/snn/test_train_L0.log