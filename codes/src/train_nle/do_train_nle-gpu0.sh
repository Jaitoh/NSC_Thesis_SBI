#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd ~/data/NSC/codes
source activate sbi

ROOT_DIR=~/data/NSC

TRAIN_FILE_NAME=train_p2
EXP_ID=L0-nle-p2-cnn
CONFIG_PRIOR=prior-3

RUN_ID=L0-nle-p2-cnn-dur3
CONFIG_DATASET=dataset-p2-dur3
CONFIG_TRAIN=train-nle-cnn

# RUN_ID=L0-nle-p2-cnn-dur3to7
# CONFIG_DATASET=dataset-p2-dur3to7
# CONFIG_TRAIN=train-nle-cnn

# RUN_ID=L0-nle-p2-cnn-dur3to11
# CONFIG_DATASET=dataset-p2-dur3to11
# CONFIG_TRAIN=train-nle-cnn

# DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
DATA_PATH=${ROOT_DIR}/data/dataset-comb
# CHECK_POINT_PATH='/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp-3dur-a1-1/model/best_model_state_dict_run0.pt'
CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_X_O=x_o-0

LOG_DIR=${ROOT_DIR}/codes/src/train_nle/logs/${EXP_ID}/${RUN_ID}
PRINT_LOG=${ROOT_DIR}/codes/src/train_nle/logs/${EXP_ID}/${RUN_ID}/${RUN_ID}.log

mkdir -p ${LOG_DIR}

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"

code ${PRINT_LOG}

nice python3 -u ${ROOT_DIR}/codes/src/train_nle/${TRAIN_FILE_NAME}.py \
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
# python3 -u ./src/train/check_log/check_log_p4.py \
#     --log_dir ${LOG_DIR} \
#     --exp_name ${RUN_ID} \
#     --num_frames 10 \
#     --duration 1000

# code ${LOG_DIR}/posterior-${RUN_ID}.gif
