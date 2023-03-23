#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 32G       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=12 ## 32 cores per task
#SBATCH --job-name=dataset_gen ## job name 
#SBATCH --output=./cluster/train_L0.out ## standard out file 

#SBATCH --gres=gpu:V100:1
#SBATCH --constraint=GPUMEM32GB

# module load amd
# module load intel

# module load anaconda3
# source activate sbi

# generate dataset
python3 -u ./src/train/train_L0.py \
--run_simulator \
--config_simulator_path './src/config/test_simulator.yaml' \
--config_dataset_path './src/config/test_dataset.yaml' \
--config_train_path './src/config/test_train.yaml' \
--log_dir './src/train/log_test' \
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

