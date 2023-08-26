#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
cd ~/data/NSC/codes
source activate sbi

ROOT_DIR="$HOME/data/NSC"

pipeline_version="nle-p2"
train_id="L0-nle-p2-cnn"

exp_id="L0-nle-p2-cnn-datav2"
# exp_id="L0-nle-p2-cnn-datav2-small-batch-tmp"
log_exp_id="nle-p2-cnn-datav2"

# use_chosen_dur=True
use_chosen_dur=$1        # 0/1
T_idx=$2                 # 0->27
iid_batch_size_theta=100 # 38GB GPU memory

# ==========
# LOG_DIR="./src/train_nle/logs/${RUN_ID}/${EXP_ID}"
LOG_DIR="${ROOT_DIR}/codes/notebook/figures/compare/posterior"
if [ $use_chosen_dur -eq 1 ]; then
    PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_T${T_idx}_chosen_dur.log"
else
    PRINT_LOG="${LOG_DIR}/${log_exp_id}_posterior_samples_T${T_idx}_all.log"
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
    >${PRINT_LOG} 2>&1

#  /home/wehe/data/NSC/codes/notebook/nle_inference-gpu-0.sh 1 1
