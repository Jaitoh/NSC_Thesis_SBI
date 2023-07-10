#!/bin/bash
### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH
### Here are the SBATCH parameters that you should always consider:

#SBATCH --array=0-99

#SBATCH --time=0-3:00:00 ## days-hours:minutes:seconds
#SBATCH --ntasks=1

#SBATCH --mem 10G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=dataset
#SBATCH --output=./cluster/uzh/dataset/feature_gen/feature-Eset0_%a.out
#SBATCH --error=./cluster/uzh/dataset/feature_gen/feature-Eset0_%a.err

# SLURM_ARRAY_TASK_ID=0

# CLUSTER=snn
CLUSTER=uzh
RUN_ID=feature-Eset0-T500v2

# DATA_PATH="/home/wehe/tmp/NSC/data/dataset/dataset_L0_eset_0_set100_T500.h5"
# DATA_PATH=/home/ubuntu/tmp/NSC/data/dataset/dataset-L0-Eset0-100sets-T500v2.h5
# FEAT_PATH=/home/ubuntu/tmp/NSC/data/dataset/feature-L0-Eset0-100sets-T500v2-C100-set${SLURM_ARRAY_TASK_ID}.h5
DATA_PATH=/home/wehe/scratch/data/dataset-L0-Eset0-100sets-T500v2.h5
FEAT_PATH=/home/wehe/scratch/data/feature/v2/feature-L0-Eset0-100sets-T500v2-C100-set${SLURM_ARRAY_TASK_ID}.h5

PRINT_DIR="./cluster/${CLUSTER}/feature/"
PRINT_LOG="./cluster/${CLUSTER}/feature/${RUN_ID}-set${SLURM_ARRAY_TASK_ID}.log"

module load anaconda3
source activate sbi

echo "print_log: ${PRINT_LOG}"
code ${PRINT_LOG}

python3 -u ./src/features/features.py \
    --data_path ${DATA_PATH} \
    --set_idx ${SLURM_ARRAY_TASK_ID} \
    --feat_path ${FEAT_PATH} >${PRINT_LOG} 2>&1

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
