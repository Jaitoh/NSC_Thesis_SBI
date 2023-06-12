#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=6-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --mem 24G
#SBATCH --cpus-per-task=4

#SBATCH --job-name=train_L0
#SBATCH --output=./cluster/uzh/train_L0/other_logs/output-p3-test.out
#SBATCH --error=./cluster/uzh/train_L0/other_logs/error-p3-test.err

CLUSTER=uzh
# PORT=6007

RUN_ID=exp-p3-test
TRAIN_FILE_NAME=train_L0

DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
# DATA_PATH=/home/wehe/scratch/data/dataset/dataset_L0_exp_set_0.h5
CONFIG_DATASET=dataset-p3-test
CONFIG_TRAIN=train-p3-test

# CHECK_POINT_PATH='/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp-3dur-a1-1/model/best_model_state_dict_run0.pt'

CONFIG_EXP=exp-set-test
CONFIG_PRIOR=prior-test
CONFIG_X_O=x_o-0
CONFIG_SIMULATOR=model-0

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

code ${PRINT_LOG}

# --run ${SLURM_ARRAY_TASK_ID} \
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
> ${PRINT_LOG} 2>&1
# debug=True\
# & tensorboard --logdir=${LOG_DIR} --port=${PORT}
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