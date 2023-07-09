#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source activate sbi

CLUSTER=win
ROOT_DIR="/c/Users/hwj/tmp/NSC"
cd ${ROOT_DIR}/codes

# RUN_ID=p5a-conv_lstm-Tv2-0
# CONFIG_PRIOR=prior-v2-0
CONFIG_DATASET=dataset-p5
CONFIG_TRAIN=train-p5-conv_lstm-mdn

RUN_ID=p5a-conv_lstm-Tv2-1
CONFIG_PRIOR=prior-v2-1

TRAIN_FILE_NAME=train_L0_p5a
# DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
# DATA_PATH="${ROOT_DIR}/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
DATA_PATH="${ROOT_DIR}/data/dataset/dataset-L0-Eset0-98sets-T500v2.h5"
# CHECK_POINT_PATH='/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp-3dur-a1-1/model/best_model_state_dict_run0.pt'
CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_X_O=x_o-0

LOG_DIR="./src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
PRINT_LOG="${LOG_DIR}/${CLUSTER}-${RUN_ID}.log"
rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"

code ${PRINT_LOG}
source activate sbi

# "D:\Program\Anaconda3\envs\sbi\python.exe"
nice /d/Program/Anaconda3/envs/sbi/python.exe -u ./src/train/${TRAIN_FILE_NAME}.py \
    hydra.run.dir=${LOG_DIR} \
    experiment_settings=${CONFIG_EXP} \
    prior=${CONFIG_PRIOR} \
    x_o=${CONFIG_X_O} \
    simulator=${CONFIG_SIMULATOR} \
    dataset=${CONFIG_DATASET} \
    train=${CONFIG_TRAIN} \
    log_dir=${LOG_DIR} \
    data_path=${DATA_PATH} \
    seed=100 \
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
