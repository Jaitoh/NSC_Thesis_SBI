#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-00:30:00 ## days-hours:minutes:seconds 
#SBATCH --mem 2G       ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1        ## Not strictly necessary because default is 1 
#SBATCH --cpus-per-task=2 ## 32 cores per task
#SBATCH --output=./cluster/build_DM_model_cy.out ## standard out file 

module load intel

module load anaconda3
source activate sbi

# build DM_Compute
echo 'build DM_compute'
cd ./src/simulator
chmod +x ./DM_compute_build.sh
./DM_compute_build.sh