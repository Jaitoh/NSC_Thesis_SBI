#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=6-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="GPUMEM16GB|GPUMEM32GB"

#SBATCH --mem 128G
#SBATCH --cpus-per-task=5

#SBATCH --job-name=train_L0
#SBATCH --output=./cluster/uzh/train_L0/other_logs/output.out
#SBATCH --error=./cluster/uzh/train_L0/other_logs/error.err

TRAIN_FILE_NAME=train_L0
CLUSTER=uzh
RUN_ID=exp_b0

# CONFIG_SIMULATOR_PATH=./src/config/test/test_simulator.yaml
# CONFIG_DATASET_PATH=./src/config/test/test_dataset.yaml
# CONFIG_TRAIN_PATH=./src/config/test/test_train.yaml

CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml
CONFIG_DATASET_PATH=./src/config/dataset/dataset_setting_0.yaml
CONFIG_TRAIN_PATH=./src/config/train/train_setting_0.yaml

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
--seed 101 \
--config_simulator_path ${CONFIG_SIMULATOR_PATH} \
--config_dataset_path ${CONFIG_DATASET_PATH} \
--config_train_path ${CONFIG_TRAIN_PATH} \
--data_path ${DATA_PATH} \
--log_dir ${LOG_DIR} \
--gpu \
-y &> ${PRINT_LOG}

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

