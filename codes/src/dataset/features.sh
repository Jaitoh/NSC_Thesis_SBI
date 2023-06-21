#!/bin/bash
### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH
### Here are the SBATCH parameters that you should always consider:

#SBATCH --array=0-99

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds
#SBATCH --ntasks=1

#SBATCH --mem 10G
#SBATCH --cpus-per-task=32

#SBATCH --job-name=dataset
#SBATCH --output=./cluster/uzh/dataset/data_gen/feature_exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/data_gen/feature_exp_set_0_%a.err

SLURM_ARRAY_TASK_ID=0

CLUSTER=snn
RUN_ID=feature_eset_0

DATA_PATH="/home/wehe/tmp/NSC/data/dataset/dataset_L0_eset_0_set100_T500.h5"
# DATA_PATH=/home/wehe/scratch/data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/${RUN_ID}_set${SLURM_ARRAY_TASK_ID}.log"

module load anaconda3
source activate sbi

echo "print_log: ${PRINT_LOG}"

python3 -u ./src/dataset/features.py \
    --data_path ${DATA_PATH} \
    --set_idx ${SLURM_ARRAY_TASK_ID} &>${PRINT_LOG}

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
# 0: 22G
# 1: 32G
