#!/bin/bash
### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH
### Here are the SBATCH parameters that you should always consider:

#SBATCH --array=1-99

#SBATCH --time=5-12:00:00 ## days-hours:minutes:seconds
#SBATCH --ntasks=1

#SBATCH --mem 8G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=dataset
#SBATCH --output=./cluster/uzh/dataset/data_gen/exp_set_0_%a.out
#SBATCH --error=./cluster/uzh/dataset/data_gen/exp_set_0_%a.err

# SLURM_ARRAY_TASK_ID=0

CLUSTER=uzh
RUN_ID=Eset0_priorV2

CONFIG_EXP=exp-set-0
CONFIG_PRIOR=prior-v2-0
CONFIG_SIMULATOR=model-0

if [ "${CLUSTER}" == "uzh" ]; then
    DATA_PATH=/home/wehe/scratch/data/dataset/v2/dataset-L0-Eset0-100sets-T500v2-part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=/home/wehe/scratch/data/dataset/v2/
else
    DATA_PATH=../data/dataset/dataset_part_${SLURM_ARRAY_TASK_ID}.h5
    DATA_DIR=../data/dataset/
fi

PRINT_DIR="./cluster/${CLUSTER}/dataset/"
PRINT_LOG="./cluster/${CLUSTER}/dataset/${RUN_ID}_${SLURM_ARRAY_TASK_ID}.log"
SEED=$((100 + SLURM_ARRAY_TASK_ID))

module load anaconda3
source activate sbi

echo "print_log: ${PRINT_LOG}"
echo "SEED: ${SEED}"

mkdir -p $DATA_DIR
mkdir -p $PRINT_DIR

python3 -u ./src/dataset/simulate_and_save.py \
    hydra.run.dir=${PRINT_DIR} \
    experiment_settings=${CONFIG_EXP} \
    prior=${CONFIG_PRIOR} \
    simulator=${CONFIG_SIMULATOR} \
    data_path=${DATA_PATH} \
    seed=${SEED} \
    log_dir=${PRINT_DIR} \
    run=${SLURM_ARRAY_TASK_ID} \
    data_path=${DATA_PATH} \
    >${PRINT_LOG} 2>&1

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
