#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 



#SBATCH --time=6-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --constraint="GPUMEM32GB"

#SBATCH --mem 200G
#SBATCH --cpus-per-task=9

#SBATCH --job-name=train_L0
#SBATCH --output=./cluster/uzh/train_L0/other_logs/output-3dur-a2.out
#SBATCH --error=./cluster/uzh/train_L0/other_logs/error-3dur-a2.err

CLUSTER=uzh
PORT=6007

RUN_ID=exp-p2-3dur-a1
TRAIN_FILE_NAME=train_L0

DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
# DATA_PATH=/home/wehe/scratch/data/dataset/dataset_L0_exp_set_0.h5
CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml
CONFIG_DATASET_PATH=./src/config/dataset/dataset-p2-0.yaml
CONFIG_TRAIN_PATH=./src/config/train/train-p2-1.yaml

LOG_DIR=/home/wehe/scratch/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
PRINT_LOG="${LOG_DIR}/${CLUSTER}-${RUN_ID}.log"
rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

module load anaconda3
source activate sbi

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"
echo "config_simulator_path: ${CONFIG_SIMULATOR_PATH}"
echo "config_dataset_path: ${CONFIG_DATASET_PATH}"
echo "config_train_path: ${CONFIG_TRAIN_PATH}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/${TRAIN_FILE_NAME}.py \
--seed 100 \
--config_simulator_path ${CONFIG_SIMULATOR_PATH} \
--config_dataset_path ${CONFIG_DATASET_PATH} \
--config_train_path ${CONFIG_TRAIN_PATH} \
--data_path ${DATA_PATH} \
--log_dir ${LOG_DIR} > ${PRINT_LOG} 2>&1
# --continue_from_checkpoint ${CHECK_POINT_PATH} \

echo 'finished simulation'

# check behavior output
python3 -u ./src/train/check_log/check_log.py \
--log_dir ${LOG_DIR} \
--exp_name ${RUN_ID} \
--num_frames 5 \
--duration 1000

echo "finished check log events"

# code ${LOG_DIR}/posterior-${RUN_ID}.gif

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

# SBATCH --array=4,5,2