#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd ~/tmp/NSC/codes
source activate sbi

ROOT_DIR="$HOME/tmp/NSC"

pipeline_version="nle-p2"
train_id="L0-nle-p2-cnn"

exp_id="L0-nle-p2-cnn-datav2"
# exp_id="L0-nle-p2-cnn-datav2-small-batch-tmp"
log_exp_id="nle-p2-cnn-datav2"

T_idx=$1 # 0->27
# use_chosen_dur=True
use_chosen_dur=$2 # True/False

# ==========
# LOG_DIR="./src/train_nle/logs/${RUN_ID}/${EXP_ID}"
LOG_DIR="${ROOT_DIR}/codes/notebook/figures/compare/posterior"
if use_chosen_dur; then
    PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_T${T_idx}_chosen_dur.log"
else
    PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_T${T_idx}_all.log"
fi

# mkdir -p ${LOG_DIR}

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
    >${PRINT_LOG} 2>&1
