#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=6-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="GPUMEM32GB"

#SBATCH --mem 100G
#SBATCH --cpus-per-task=10

#SBATCH --job-name=train_L0
#SBATCH --output=./cluster/uzh/train_L0/other_logs/output-c0-sub5.out
#SBATCH --error=./cluster/uzh/train_L0/other_logs/error-c0-sub5.err

TRAIN_FILE_NAME=train_L0
CLUSTER=uzh

CONFIG_TRAIN_PATH=./src/config/train/train_setting_0.yaml
CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml

# ---
# RUN_ID=exp-c0-sub0
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub0.yaml
# RUN_ID=exp-c0-sub1
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub1.yaml
# RUN_ID=exp-c0-sub2
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub2.yaml
# RUN_ID=exp-c0-sub3
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub3.yaml
# RUN_ID=exp-c0-sub4
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub4.yaml
RUN_ID=exp-c0-sub5
CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-1-sub5.yaml

# RUN_ID=exp_b1
# RUN_ID=exp_b2_0
# RUN_ID=exp_b3_0
# RUN_ID=exp_b0_2
# RUN_ID=exp_b2_1_
# RUN_ID=exp_b2_2_
# RUN_ID=exp_b3_1_
# RUN_ID=exp_b3_2_
# RUN_ID=exp_b0_3_

# ---
# RUN_ID=exp-b0-4
# RUN_ID=exp-b2-3
# RUN_ID=exp-b2-4
# RUN_ID=exp-b3-3
# RUN_ID=exp-b3-4
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_shuffle_1.yaml
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_0.yaml
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_1.yaml
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_2.yaml # b0_2
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_0_1_.yaml # b2_1
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_0_2_.yaml # b2_2
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_1_1_.yaml # b3_1
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_summary_1_2_.yaml # b3_2
# CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0_3_.yaml # b0_3
# ---
# RUN_ID=exp-b0-2-contd0
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-2.yaml # RUN_ID=exp-b0-2-contd0
# CHECK_POINT_PATH='/home/wehe/scratch/train/logs/train_L0/exp_b0_2/model/best_model_state_dict_run0.pt'

# RUN_ID=exp-b2-1-contd0
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-0-1.yaml # RUN_ID=exp-b2-1-contd0
# CHECK_POINT_PATH='/home/wehe/scratch/train/logs/train_L0/exp_b2_1/model/best_model_state_dict_run0.pt'

# RUN_ID=exp-b2-2-contd0
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-0-2.yaml # RUN_ID=exp-b2-2-contd0
# CHECK_POINT_PATH='/home/wehe/scratch/train/logs/train_L0/exp_b2_2/model/best_model_state_dict_run0.pt'

# RUN_ID=exp-b3-2-contd0
# CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-1-2.yaml # RUN_ID=exp-b3-2-contd0
# CHECK_POINT_PATH='/home/wehe/scratch/train/logs/train_L0/exp_b3_2/model/best_model_state_dict_run0.pt'

# RUN_ID=exp-b0-3-contd0
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-3.yaml # RUN_ID=exp-b0-3-contd0
# CHECK_POINT_PATH='/home/wehe/scratch/train/logs/train_L0/exp_b0_3/model/best_model_state_dict_run0.pt'
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-4.yaml # RUN_ID=exp-b0-4
# CHECK_POINT_PATH=''
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-0-3.yaml # RUN_ID=exp-b2-3
# CHECK_POINT_PATH=''
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-0-4.yaml # RUN_ID=exp-b2-4
# CHECK_POINT_PATH=''
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-1-3.yaml # RUN_ID=exp-b3-3
# CHECK_POINT_PATH=''
# # CONFIG_DATASET_PATH=./src/config/dataset/dataset-setting-0-summary-1-4.yaml # RUN_ID=exp-b3-4
# CHECK_POINT_PATH=''

if [ "${CLUSTER}" == "uzh" ]; then
    LOG_DIR=/home/wehe/scratch/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
    # DATA_PATH=/home/wehe/scratch/data/dataset/dataset_L0_exp_set_0.h5
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    module load anaconda3
    source activate sbi
else
    LOG_DIR="./src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
fi

PRINT_LOG="./cluster/${CLUSTER}/${TRAIN_FILE_NAME}/${RUN_ID}.log"
mkdir -p ./cluster/${CLUSTER}/${TRAIN_FILE_NAME}/other_logs

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
--log_dir ${LOG_DIR} \
--gpu \
-y &> ${PRINT_LOG}
# --continue_from_checkpoint ${CHECK_POINT_PATH} \


echo 'finished simulation'

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952
# squeue -u $USER
# scancel --user=wehe
# squeue -u $USER
# squeue -u $USER

# SBATCH --gres=gpu:T4:1
# SBATCH --gres=gpu:V100:1
# SBATCH --gres=gpu:A100:1

# 829580_* job - array=5 ?
# 829605_* job - array=3-4 T4 requested
# 829576_* job - array=0-1 V100
# 829500_* job - array=2 A100
# SBATCH --array=0-5

# ./src/train/do_train_uzh.sh 
# SBATCH --constraint="GPUMEM16GB|GPUMEM32GB"
