#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ROOT_DIR=~/data/NSC
cd ${ROOT_DIR}/codes
source activate sbi

RUN_ID=p4a-F1345-cnn-maf3-B
CONFIG_DATASET=dataset-p4-F1345-size0

CONFIG_PRIOR=prior-3
CONFIG_TRAIN=train-p4-cnn-maf3

TRAIN_FILE_NAME=train_L0_p4a
DATA_PATH="${ROOT_DIR}/data/dataset/feature-L0-Eset0-100sets-T60-C100-B20.h5"
CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_X_O=x_o-0

LOG_DIR="${ROOT_DIR}/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
PRINT_LOG="${LOG_DIR}/${RUN_ID}.log"
# rm -r ${LOG_DIR}/events.out.tfevents*
mkdir -p ${LOG_DIR}

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"

code ${PRINT_LOG}

nice python3 -u ./src/train/${TRAIN_FILE_NAME}.py \
    hydra.run.dir=${LOG_DIR} \
    experiment_settings=${CONFIG_EXP} \
    prior=${CONFIG_PRIOR} \
    x_o=${CONFIG_X_O} \
    simulator=${CONFIG_SIMULATOR} \
    dataset=${CONFIG_DATASET} \
    train=${CONFIG_TRAIN} \
    log_dir=${LOG_DIR} \
    data_path=${DATA_PATH} \
    seed=42 \
    debug=False \
    >${PRINT_LOG} 2>&1
# debug=True\
# & tensorboard --logdir=${LOG_DIR} --port=${PORT}
# --continue_from_checkpoint ${CHECK_POINT_PATH} \

echo "finished training"

# check behavior output
python3 -u ./src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 10 \
    --duration 1000

code ${LOG_DIR}/posterior-${RUN_ID}.gif
