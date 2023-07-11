#!/bin/bash

CLUSTER=t4
TRAIN_FILE_NAME=train_L0_p5a

# RUN_ID=p4-5Fs-1D-mh_gru-mdn
# RUN_ID=p4-5Fs-2D-mh_gru-mdn
# RUN_ID=p4-5Fs-1D-gru-mdn
RUN_ID=p4-5Fs-2D-mh_gru-mdn-ctd0
RUN_ID=p4-5Fs-1D-mh_gru-mdn-ctd0
RUN_ID=p4-5Fs-1D-gru-mdn-ctd0
RUN_ID=p5-gru3
RUN_ID=p5-conv_lstm
RUN_ID=p5-conv_transformer
RUN_ID=p5a-conv_lstm-Tv2-0

if [ "${CLUSTER}" == "t4" ]; then
    LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
elif [ "${CLUSTER}" == "snn" ]; then
    LOG_DIR="/home/wehe/tmp/NSC/codes/src/train/logs//${TRAIN_FILE_NAME}/${RUN_ID}"
fi

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
code ${LOG_DIR}/posterior-${RUN_ID}.png

echo 'finished check log events'
