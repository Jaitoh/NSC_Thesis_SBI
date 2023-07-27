#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
cd ~/tmp/NSC/codes
source activate sbi

CLUSTER=snn
ROOT_DIR="/home/wehe/tmp/NSC"

RUN_ID=L0-nle-cnn-dur3-online-copy

LOG_DIR="./src/train/logs/${RUN_ID}"
PRINT_LOG="./src/train/logs/posterior-${RUN_ID}.log"
# rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"

code ${PRINT_LOG}

nice python3 -u ${ROOT_DIR}/codes/src/train_nle/get_posterior.py \
    hydra.run.dir=${LOG_DIR} \
    data_path=${ROOT_DIR}/data/dataset-comb \
    >${PRINT_LOG} 2>&1
