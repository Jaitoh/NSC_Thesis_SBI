#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
cd ~/tmp/NSC/codes
source activate sbi

CLUSTER=snn
ROOT_DIR="$HOME/tmp/NSC"

RUN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-offline
CONFIG_DATASET=dataset-nle-cnn-dur3-offline
CONFIG_POST=posterior-dur3-t0 #snn
CONFIG_POST=posterior-dur3-t1 #snn
CONFIG_POST=posterior-dur3-t2 #snn

CONFIG_POST=posterior-dur3-t3 #snn
# CONFIG_POST=posterior-dur3-v0 #snn
# CONFIG_POST=posterior-dur3-v1 #snn
# CONFIG_POST=posterior-dur3-v2 #t4-1
# CONFIG_POST=posterior-dur3-s2 #t4-2
CONFIG_POST=posterior-dur3-s3 #snn

LOG_DIR="./src/train_nle/logs/${RUN_ID}/${EXP_ID}"
PRINT_LOG="${LOG_DIR}/posterior-${RUN_ID}-${CONFIG_POST}.log"
# rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"

code ${PRINT_LOG}

SCRIPT_PATH=${ROOT_DIR}/codes/src/train_nle/get_posterior.py
nice python3 -u ${SCRIPT_PATH} \
    hydra.run.dir=${LOG_DIR} \
    log_dir=${LOG_DIR} \
    data_path=${ROOT_DIR}/data/dataset-comb \
    dataset=${CONFIG_DATASET} \
    posterior=${CONFIG_POST} \
    >${PRINT_LOG} 2>&1
