#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 24G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=train_L0_rnn_npe_data
#SBATCH --output=./cluster/uzh/train_L0_rnn_npe_data/train_logs/a0.out
#SBATCH --error=./cluster/uzh/train_L0_rnn_npe_data/train_logs/a0.err


RUN_ID=a0
TRAIN_FILE_NAME=train_L0_rnn_npe_data

CLUSTER=snn
CONFIG_SIMULATOR_PATH=./src/config/test/test_simulator.yaml
CONFIG_DATASET_PATH=./src/config/test/test_dataset.yaml
CONFIG_TRAIN_PATH=./src/config/test/test_train.yaml

LOG_DIR=./src/train/logs/$TRAIN_FILE_NAME/$RUN_ID
# LOG_DIR=/home/wehe/scratch/train/logs/$TRAIN_FILE_NAME/$RUN_ID
PRINT_LOG=./cluster/$CLUSTER/$TRAIN_FILE_NAME/train_logs/$RUN_ID.log

# module load anaconda3
# source activate sbi
# # module load t4
# module load gpu
# module load cuda
echo $LOG_DIR

python3 -u ./src/train/$TRAIN_FILE_NAME.py \
--seed 100 \
--config_simulator_path $CONFIG_SIMULATOR_PATH \
--config_dataset_path $CONFIG_DATASET_PATH \
--config_train_path $CONFIG_TRAIN_PATH \
--log_dir $LOG_DIR \
--gpu \
-y > $PRINT_LOG

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
