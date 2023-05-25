#!/bin/bash 

CLUSTER=uzh
RUN_ID=exp-p2-3dur-a1
TRAIN_FILE_NAME=train_L0

if [ "${CLUSTER}" == "uzh" ]; then
    LOG_DIR=/home/wehe/scratch/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    # module load anaconda3
    # source activate sbi
fi

echo "log_dir: ${LOG_DIR}"

# --run ${SLURM_ARRAY_TASK_ID} \
/home/wehe/data/conda/envs/sbi/bin/python3 -u ./src/train/check_log/check_log.py \
--log_dir ${LOG_DIR} \
--exp_name ${RUN_ID} \
--num_frames 5 \
--duration 1000

# open files
# code ${LOG_DIR}/training_curve_.png
code ${LOG_DIR}/posterior-${RUN_ID}.gif

echo 'finished check log events'