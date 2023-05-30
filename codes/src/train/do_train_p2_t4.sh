#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
cd ~/tmp/NSC/codes
source activate sbi

CLUSTER=t4
PORT=9906

RUN_ID=exp-p2-3dur-a5
TRAIN_FILE_NAME=train_L0

DATA_PATH="/mnt/data/dataset_L0_exp_set_0.h5"
# DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml
CONFIG_DATASET_PATH=./src/config/dataset/dataset-p2-2.yaml
CONFIG_TRAIN_PATH=./src/config/train/train-p2-2.yaml

# CHECK_POINT_PATH='/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp-3dur-a1-1/model/best_model_state_dict_run0.pt'

# PRINT_LOG="./cluster/${CLUSTER}/${TRAIN_FILE_NAME}/output_logs/${RUN_ID}.log"
LOG_DIR="./src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
PRINT_LOG="${LOG_DIR}/${CLUSTER}-${RUN_ID}.log"
rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"
echo "config_simulator_path: ${CONFIG_SIMULATOR_PATH}"
echo "config_dataset_path: ${CONFIG_DATASET_PATH}"
echo "config_train_path: ${CONFIG_TRAIN_PATH}"

code ${PRINT_LOG}

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/${TRAIN_FILE_NAME}.py \
--seed 100 \
--config_simulator_path ${CONFIG_SIMULATOR_PATH} \
--config_dataset_path ${CONFIG_DATASET_PATH} \
--config_train_path ${CONFIG_TRAIN_PATH} \
--data_path ${DATA_PATH} \
--log_dir ${LOG_DIR} > ${PRINT_LOG} 2>&1 & tensorboard --logdir=${LOG_DIR} --port=${PORT}
# --continue_from_checkpoint ${CHECK_POINT_PATH} \

echo "finished simulation"

# check behavior output
python3 -u ./src/train/check_log/check_log.py \
--log_dir ${LOG_DIR} \
--exp_name ${RUN_ID} \
--num_frames 10 \
--duration 1000

# code ${LOG_DIR}/training_curve_.png
# code ${LOG_DIR}/posterior_shuffled.gif
code ${LOG_DIR}/posterior-${RUN_ID}.gif