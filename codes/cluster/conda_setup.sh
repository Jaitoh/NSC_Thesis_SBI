#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-00:20:00 ## days-hours:minutes:seconds 
#SBATCH --mem 500M       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=1 ## 32 cores per task
#SBATCH --job-name=conda_setup ## job name 
#SBATCH --output=./cluster/conda_setup.out ## standard out file 

module load intel

# environment setup
echo 'start environment setup' 
date
module load anaconda3
conda create -n sbi python=3.7 && source activate sbi
source activate sbi
pip install sbi
pip install cython
pip install h5py
echo 'finished environment setup'
date 
