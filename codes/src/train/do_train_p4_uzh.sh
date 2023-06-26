#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 

#SBATCH --time=6-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=10

#SBATCH --job-name=train_L0_p4
#SBATCH --output=./cluster/uzh/train_L0_p4/other_logs/output-p4.out
#SBATCH --error=./cluster/uzh/train_L0_p4/other_logs/error-p4.err

CLUSTER=uzh
module load anaconda3
source activate sbi

# RUN_ID=p4-F1-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F1-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F2-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F2-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F3-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F3-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

# RUN_ID=p4-F4-1D-gru3-mdn
# CONFIG_DATASET=dataset-p4-F4-1D
# CONFIG_TRAIN=train-p4-gru3-mdn

RUN_ID=p4-F5-1D-gru3-mdn
CONFIG_DATASET=dataset-p4-F5-1D
CONFIG_TRAIN=train-p4-gru3-mdn

TRAIN_FILE_NAME=train_L0_p4
DATA_PATH="/home/wehe/data/NSC/data/dataset/feature-L0-Eset0-100sets-T500-C100.h5"

CONFIG_SIMULATOR=model-0
CONFIG_EXP=exp-set-0
CONFIG_PRIOR=prior-0
CONFIG_X_O=x_o-0

LOG_DIR="/home/wehe/scratch/train/logs/${TRAIN_FILE_NAME}/${RUN_ID}"
PRINT_LOG="${LOG_DIR}/${CLUSTER}-${RUN_ID}.log"
rm -r ${LOG_DIR}
mkdir -p ${LOG_DIR}

echo "file name: ${TRAIN_FILE_NAME}"
echo "log_dir: ${LOG_DIR}"
echo "print_log: ${PRINT_LOG}"
echo "data_path: ${DATA_PATH}"

# code ${PRINT_LOG}

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
    debug=False \
    >${PRINT_LOG} 2>&1
# continue_from_checkpoint=${CHECK_POINT_PATH} \
# debug=True\
# & tensorboard --logdir=${LOG_DIR} --port=${PORT}

echo "finished training"

# check behavior output
python3 -u ./src/train/check_log/check_log_p4.py \
    --log_dir ${LOG_DIR} \
    --exp_name ${RUN_ID} \
    --num_frames 10 \
    --duration 1000

code ${LOG_DIR}/posterior-${RUN_ID}.gif


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