#!/bin/bash

CLUSTER=snn
TRAIN_FILE_NAME=train_L0_p4
# RUN_ID=p4-5Fs-1D-mh_gru-mdn
# RUN_ID=p4-5Fs-2D-mh_gru-mdn
# RUN_ID=p4-5Fs-1D-gru-mdn
RUN_ID=p4-5Fs-2D-mh_gru-mdn-ctd0
RUN_ID=p4-5Fs-1D-mh_gru-mdn-ctd0
RUN_ID=p4-5Fs-1D-gru-mdn-ctd0
RUN_ID=p4-5Fs-1D-gru3-mdn-ctd
RUN_ID=p4-3Fs-1D-gru3-mdn
RUN_ID=p4-4Fs-1D-gru3-mdn
RUN_ID=p4-F1-1D-gru3-mdn
RUN_ID=p4-F2-1D-gru3-mdn
RUN_ID=p4-F3-1D-gru3-mdn
RUN_ID=p4-F4-1D-gru3-mdn
RUN_ID=p4-F5-1D-gru3-mdn

TRAIN_FILE_NAME=train_L0_p5a
RUN_ID=p5a-conv_lstm

cd ~/tmp/NSC/codes
source activate sbi

LOG_DIR=$HOME/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
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
