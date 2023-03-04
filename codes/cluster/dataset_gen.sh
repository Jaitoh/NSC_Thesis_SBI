#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-01:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 16G       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=48 ## 32 cores per task
#SBATCH --job-name=dataset_gen ## job name 
#SBATCH --output=./cluster/dataset_gen.out ## standard out file 

module load intel

# environment setup
echo 'start environment setup' 
date
module load anaconda3
conda env create -f environment.yml
source activate sbi
echo 'finished environment setup'
date 

# call the code
echo 'start simulation'
date 
python3 ./src/data_generator/dataset_for_training.py
echo 'finished simulation'
date 