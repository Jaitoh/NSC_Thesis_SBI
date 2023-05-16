#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=18

#SBATCH --job-name=sim_data_for_round_0
#SBATCH --output=./cluster/uzh/sim_data_for_round_0/other_logs/a0_%a.out
#SBATCH --error=./cluster/uzh/sim_data_for_round_0/other_logs/a0_%a.err

export CUDA_VISIBLE_DEVICES=1
CLUSTER=snn
TRAIN_FILE_NAME=train_L0
# RUN_ID=exp-b2-2-contd0
# RUN_ID=exp-d0-net3
RUN_ID=exp-dur3-e0

if [ "${CLUSTER}" == "uzh" ]; then
    LOG_DIR=/home/wehe/scratch/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    module load anaconda3
    source activate sbi
fi

if [ "${CLUSTER}" == "snn" ]; then
    LOG_DIR="/home/wehe/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    cd ~/tmp/NSC/codes
    source activate sbi 
fi

if [ "${CLUSTER}" == "t4" ]; then
    LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    cd ~/tmp/NSC/codes
    source activate sbi 
fi

if [ "${CLUSTER}" == "sensors" ]; then
    LOG_DIR="/home/wenjie/NSC/codes/src/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
    DATA_PATH="../data/dataset/dataset_L0_exp_set_0.h5"
    cd ~/tmp/NSC/codes
    source activate sbi 
fi

echo "log_dir: ${LOG_DIR}"

# --run ${SLURM_ARRAY_TASK_ID} \
python3 -u ./src/train/check_log.py \
--log_dir ${LOG_DIR} \
--num_rows 1 \
--plot_posterior \
--exact_epoch


echo 'finished check log events'

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