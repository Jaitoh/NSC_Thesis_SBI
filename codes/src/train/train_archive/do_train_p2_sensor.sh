#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 96G
#SBATCH --cpus-per-task=18

#SBATCH --job-name=sim_data_for_round_0
#SBATCH --output=./cluster/uzh/sim_data_for_round_0/other_logs/a0_%a.out
#SBATCH --error=./cluster/uzh/sim_data_for_round_0/other_logs/a0_%a.err

export CUDA_VISIBLE_DEVICES=1
cd ~/NSC/codes
# source activate sbi

TRAIN_FILE_NAME=train_L0
CLUSTER=sensors

CONFIG_TRAIN_PATH=./src/config/train/train-setting-1.yaml
CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml

# RUN_ID=exp-c0-sub1
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub1.yaml

RUN_ID=exp-c0-sub3
CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub3.yaml

# RUN_ID=exp-c0-sub4
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub4.yaml

# RUN_ID=exp-c0-sub2
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub2.yaml

# RUN_ID=exp-c0-sub5
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub5.yaml

LOG_DIR="./src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"

PRINT_LOG="./cluster/${CLUSTER}/${TRAIN_FILE_NAME}/output_logs/${RUN_ID}.log"
mkdir -p ./cluster/${CLUSTER}/${TRAIN_FILE_NAME}/output_logs

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/${TRAIN_FILE_NAME}.py \
--seed 100 \
--config_simulator_path ${CONFIG_SIMULATOR_PATH} \
--config_dataset_path ${CONFIG_DATASET_PATH} \
--config_train_path ${CONFIG_TRAIN_PATH} \
--data_path ${DATA_PATH} \
--log_dir ${LOG_DIR} &> ${PRINT_LOG}
# --gpu \
# -y &> ${PRINT_LOG}
# --continue_from_checkpoint ${CHECK_POINT_PATH} \

echo 'finished simulation'

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952

# SBATCH --gres=gpu:T4:1
# SBATCH --gres=gpu:V100:1
# SBATCH --gres=gpu:A100:1
# SBATCH --array=0-49

# cd ~/tmp/NSC/codes/
# conda activate sbi
# ./src/train/do_train_snn.sh