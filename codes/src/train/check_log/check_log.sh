#!/bin/bash

RUN_ID=exp-p3-3dur-a0
TRAIN_FILE_NAME=train_L0

# LOG_DIR=$HOME/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
RUN_ID=p5a-conv_lstm
LOG_DIR=/home/ubuntu/tmp/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_lstm-tmp

cd ~/tmp/NSC/codes
source activate sbi
echo "log_dir: ${LOG_DIR}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 10 \
    --duration 1000

# open files
code ${LOG_DIR}/posterior-${RUN_ID}.gif

echo 'finished check log events'
