#!/bin/bash 

CLUSTER=t4
RUN_ID=exp-p3-3dur-a0
TRAIN_FILE_NAME=train_L0

if [ "${CLUSTER}" == "t4" ]; then
    LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    cd ~/tmp/NSC/codes
    source activate sbi 
fi

echo "log_dir: ${LOG_DIR}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/check_log/check_log.py \
--log_dir ${LOG_DIR} \
--exp_name ${RUN_ID} \
--num_frames 5 \
--duration 1000

# open files
code ${LOG_DIR}/posterior-${RUN_ID}.gif

echo 'finished check log events'