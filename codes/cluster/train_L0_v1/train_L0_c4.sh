#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 12G       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=16 ## 32 cores per task
#SBATCH --job-name=dataset_gen ## job name 
#SBATCH --output=./cluster/train_L0/train_L0_c4.out ## standard out file 

# module load amd
# module load intel

module load anaconda3
source activate sbi

# generate dataset
# --run_simulator \
python3 -u ./src/train/train_L0.py \
--config_simulator_path './src/config/simulator_Ca_Pb_Ma.yaml' \
--config_dataset_path './src/config/dataset_Sb1_suba0_Rb0.yaml' \
--config_train_path './src/config/train_Ta1.yaml' \
--log_dir './src/train/logs/log-simulator_Ca_Pb_Ma-dataset_Sb1_suba0_Rb0-train_Ta1' \
-y > ./cluster/train_L0/train_L0_c4.log
# --gpu \

echo 'finished simulation'

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952
# squeue -u $USER
# scancel --user=wehe
# squeue -u $USER
# squeue -u $USER

