#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 32G       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=12 ## 32 cores per task
#SBATCH --job-name=dataset_gen ## job name 
#SBATCH --output=./cluster/dataset_gen.out ## standard out file 

# module load amd
# module load intel

# tmux
# conda activate sbi
# ./cluster/snn/build_DM_model_cy.sh
# generate dataset
echo 'start generating dataset'
python3 ./src/data_generator/dataset_generator.py
echo 'finished simulation'