#!/bin/bash

RUN_ID=p4-5Fs-1D-gru-mdn
TRAIN_FILE_NAME=train_L0_p4

LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
cd ~/tmp/NSC/codes
source activate sbi

echo "log_dir: ${LOG_DIR}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 5 \
    --duration 1000

# open files
code ${LOG_DIR}/posterior-${RUN_ID}.gif

echo 'finished check log events'
