#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1       

#SBATCH --mem 32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1

#SBATCH --job-name=train_L0_rnn_a2
#SBATCH --output=./cluster/uzh/train_L0_rnn/train_logs/train_L0_rnn_a2.out

# module load amd

module load intel
module load anaconda3
source activate sbi
# module load v100
# module load gpu
# module load cuda


RUN_ID=a2
TRAIN_FILE_NAME=train_L0_rnn
CLUSTER=uzh
CONFIG_SIMULATOR_PATH=./src/config/test/test_simulator_2.yaml
CONFIG_DATASET_PATH=./src/config/test/test_dataset.yaml
CONFIG_TRAIN_PATH=./src/config/test/test_train.yaml

JOB_NAME=$TRAIN_FILE_NAME_$RUN_ID
OUTPUT_FILE=./cluster/$CLUSTER/$TRAIN_FILE_NAME/train_logs/$JOB_NAME.out
LOG_DIR=./src/train/logs/$TRAIN_FILE_NAME/$RUN_ID


python3 -u ./src/train/$TRAIN_FILE_NAME.py \
--seed 100 \
--config_simulator_path $CONFIG_SIMULATOR_PATH \
--config_dataset_path $CONFIG_DATASET_PATH \
--config_train_path $CONFIG_TRAIN_PATH \
--log_dir $LOG_DIR \
--gpu \
-y

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
# sbatch ./cluster/uzh/train_L0_rnn/train_L0_rnn_a2.sh 