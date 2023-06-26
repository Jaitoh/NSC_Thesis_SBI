#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
cd ~/tmp/NSC/codes
source activate sbi

CLUSTER=snn
# PORT=9906

# RUN_ID=p4-5Fs-1D-mh_gru-mdn
# RUN_ID=p4-5Fs-1D-mh_gru-mdn-ctd0
# RUN_ID=p4-5Fs-1D-lstm3-mdn
# RUN_ID=p4-5Fs-1D-gru3-mdn-ctd
# RUN_ID=p4-3Fs-1D-gru3-mdn
# RUN_ID=p4-4Fs-1D-gru3-mdn

# CONFIG_DATASET=dataset-p4-5Fs-1D-ctd0
# CONFIG_TRAIN=train-p4-mh_gru-mdn
# CONFIG_DATASET=dataset-p4-5Fs-1D
# CONFIG_DATASET=dataset-p4-3Fs-1D
# CONFIG_DATASET=dataset-p4-4Fs-1D
# CONFIG_TRAIN=train-p4-lstm3-mdn
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F1-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F1-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F2-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F2-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F3-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F3-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F4-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F4-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

RUN_ID=p4-F5-1D-gru3-mdn
CONFIG_DATASET=dataset-p4-F5-1D
CONFIG_TRAIN=train-p4-gru3-mdn

# CHECK_POINT_PATH="/home/wehe/tmp/NSC/codes/src/train/logs/train_L0_p4/p4-5Fs-1D-gru3-mdn/model/model_check_point.pt"

TRAIN_FILE_NAME=train_L0_p4
# DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
# DATA_PATH="/home/ubuntu/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"
DATA_PATH="/home/wehe/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"

CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_PRIOR=prior-0
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

python3 -u ./src/train/${TRAIN_FILE_NAME}.py \
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
# continue_from_checkpoint=${CHECK_POINT_PATH} \
# debug=True\
# & tensorboard --logdir=${LOG_DIR} --port=${PORT}

echo "finished training"

# check behavior output
python3 -u ./src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 10 \
    --duration 1000

code ${LOG_DIR}/posterior-${RUN_ID}.gif
