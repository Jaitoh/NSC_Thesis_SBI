#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
ROOT_DIR=~/data/NSC
cd ${ROOT_DIR}/codes
source activate sbi

# RUN_ID=p5a-conv_net-Tv2
# CONFIG_PRIOR=prior-v2-3
# CONFIG_DATASET=dataset-p5
# CONFIG_TRAIN=train-p5-conv_net-mdn

# aug-06
# RUN_ID=p5a-conv_lstm-Tv2
# CONFIG_PRIOR=prior-v2-3
# CONFIG_DATASET=dataset-p5
# CONFIG_TRAIN=train-p5-conv_lstm-mdn

# aug-14
# RUN_ID=p5a-conv_lstm-corr_conv
# CONFIG_PRIOR=prior-3
# CONFIG_DATASET=dataset-p5
# CONFIG_TRAIN=train-p5-conv_lstm-mdn

RUN_ID=p5a-conv_lstm-corr_conv-B2
DATA_PATH="${ROOT_DIR}/data/dataset/dataset-L0-Eset0-20sets-T60-B20-v2.h5"

TRAIN_FILE_NAME=train_L0_p5a
CONFIG_PRIOR=prior-3
CONFIG_DATASET=dataset-p5-B
CONFIG_TRAIN=train-p5-conv_lstm-mdn
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

nice python3 -u ${ROOT_DIR}/codes/src/train/${TRAIN_FILE_NAME}.py \
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
python3 -u ${ROOT_DIR}/codes/src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 10 \
    --duration 1000

code ${LOG_DIR}/posterior-${RUN_ID}.gif
