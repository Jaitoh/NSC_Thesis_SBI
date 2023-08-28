#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
cd ~/data/NSC/codes
source activate sbi

ROOT_DIR="$HOME/data/NSC"

# pipeline_version="nle-p2"
# train_id="L0-nle-p2-cnn"

# # exp_id="L0-nle-p2-cnn-datav2"
# exp_id="L0-nle-p2-cnn-datav2-small-batch-newLoss-tmp2"
# log_exp_id="nle-p2-cnn-datav2-newLoss"

# p3 version use balanced training/validation data
pipeline_version="nle-p3"
train_id="L0-nle-p3-cnn"

exp_id="L0-nle-p3-cnn-newLoss-tmp-2"
log_exp_id="nle-p3-cnn-newLoss"

# use_chosen_dur=True
num_samples=2000
iid_batch_size_theta=100 # GB GPU memory
use_chosen_dur=$1        # 0/1 -> 1: use chosen dur, 0: use all dur
# T_idx=$2                 # 0->27
START_T_IDX=$2
END_T_IDX=$3

for T_idx in $(seq $START_T_IDX $END_T_IDX); do

    # ==========
    # LOG_DIR="./src/train_nle/logs/${RUN_ID}/${EXP_ID}"
    LOG_DIR="${ROOT_DIR}/codes/notebook/figures/nle/log"
    if [ $use_chosen_dur -eq 1 ]; then
        PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_chosen_dur_T${T_idx}.log"
    else
        PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_all_dur_T${T_idx}.log"
    fi

    # mkdir -p ${LOG_DIR}
    echo "use_chosen_dur ${use_chosen_dur}, T_idx ${T_idx}"
    echo "log_dir: ${LOG_DIR}"
    echo "print_log: ${PRINT_LOG}"

    code ${PRINT_LOG}
    SCRIPT_PATH=${ROOT_DIR}/codes/notebook/nle_inference.py
    nice python3 -u ${SCRIPT_PATH} \
        --pipeline_version ${pipeline_version} \
        --train_id ${train_id} \
        --exp_id ${exp_id} \
        --log_exp_id ${log_exp_id} \
        --use_chosen_dur ${use_chosen_dur} \
        --T_idx ${T_idx} \
        --iid_batch_size_theta ${iid_batch_size_theta} \
        --num_samples ${num_samples} \
        >${PRINT_LOG} 2>&1

    #  /home/wehe/data/NSC/codes/notebook/nle_inference-gpu-0.sh 1 1
done
