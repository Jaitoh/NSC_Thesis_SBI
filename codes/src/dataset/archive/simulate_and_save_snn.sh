#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --array=0-49

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --mem 96G
#SBATCH --cpus-per-task=18

#SBATCH --job-name=dataset
#SBATCH --output=./cluster/uzh/dataset/other_logs/exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/other_logs/exp_set_0_%a.err

SLURM_ARRAY_TASK_ID=$1

CLUSTER=snn
RUN_ID=exp_set_0

CONFIG_SIMULATOR_PATH=./src/config/simulator/exp_set_0.yaml
CONFIG_DATASET_PATH=./src/config/dataset/default.yaml
CONFIG_TRAIN_PATH=./src/config/train/default.yaml

if [ "${CLUSTER}" == "uzh" ]; then
    DATA_PATH=/home/wehe/scratch/data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=/home/wehe/scratch/data/dataset/
else
    DATA_PATH=../data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=../data/dataset/
fi

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/${RUN_ID}_${SLURM_ARRAY_TASK_ID}.log"
SEED=$((100 + SLURM_ARRAY_TASK_ID))

# module load anaconda3
# source activate sbi

echo "print_log: ${PRINT_LOG}"
echo "SEED: ${SEED}"

mkdir -p $DATA_DIR
mkdir -p $PRINT_DIR

python3 -u ./src/dataset/simulate_and_save.py \
--seed ${SEED} \
--config_simulator_path ${CONFIG_SIMULATOR_PATH} \
--config_dataset_path ${CONFIG_DATASET_PATH} \
--config_train_path ${CONFIG_TRAIN_PATH} \
--log_dir ${PRINT_DIR} \
--run ${SLURM_ARRAY_TASK_ID} \
--data_path ${DATA_PATH} &> ${PRINT_LOG}

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