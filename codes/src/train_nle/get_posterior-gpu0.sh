#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd ~/tmp/NSC/codes
source activate sbi

ROOT_DIR="$HOME/tmp/NSC"

# ===== p1 =====
SCRIPT_NAME=get_posterior.py
RUN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-offline
CONFIG_DATASET=dataset-nle-cnn-dur3-offline
# CONFIG_POST=posterior-dur3-t0 #t4-1
# CONFIG_POST=posterior-dur3-t1 #t4-1

CONFIG_POST=posterior-dur3-v0 #t4-2
# CONFIG_POST=posterior-dur3-v1 #t4-2

# CONFIG_POST=posterior-dur3-t2 #snn
# CONFIG_POST=posterior-dur3-t3 #snn
# CONFIG_POST=posterior-dur3-v2 #snn
# CONFIG_POST=posterior-dur3-s2 #snn
# CONFIG_POST=posterior-dur3-s3 #snn

# ===== p2 =====
SCRIPT_NAME=get_posterior_p2.py
RUN_ID=L0-nle-p2-cnn
EXP_ID=L0-nle-p2-cnn-dur3
CONFIG_DATASET=dataset-p2-dur3
CONFIG_TRAIN=train-nle-cnn
CONFIG_POST=posterior-p2-t0 #snn-0
CONFIG_POST=posterior-p2-t1 #t4-1
CONFIG_POST=posterior-p2-t2 #t4-2
CONFIG_POST=posterior-p2-t3 #t4-2

DATA_PATH=${ROOT_DIR}/data/dataset-comb
CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_X_O=x_o-0

# ==========
LOG_DIR="./src/train_nle/logs/${RUN_ID}/${EXP_ID}"
PRINT_LOG="${LOG_DIR}/posterior/posterior-${RUN_ID}-${CONFIG_POST}.log"
# rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"

code ${PRINT_LOG}

SCRIPT_PATH=${ROOT_DIR}/codes/src/train_nle/$SCRIPT_NAME
nice python3 -u ${SCRIPT_PATH} \
    hydra.run.dir=${LOG_DIR} \
    log_dir=${LOG_DIR} \
    data_path=${ROOT_DIR}/data/dataset-comb \
    dataset=${CONFIG_DATASET} \
    posterior=${CONFIG_POST} \
    >${PRINT_LOG} 2>&1
