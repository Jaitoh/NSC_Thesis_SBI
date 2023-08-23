#!/bin/bash
### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH
### Here are the SBATCH parameters that you should always consider:

#SBATCH --array=0-9

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds
#SBATCH --ntasks=1

#SBATCH --mem 16G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=gen_dataset_comb_13
#SBATCH --output=./cluster/uzh/prior_sim/gen_dataset_comb_13_%a.out
#SBATCH --error=./cluster/uzh/prior_sim/gen_dataset_comb_13_%a.err

module load anaconda3
source activate sbi

nice python3 -u $HOME/tmp/NSC/codes/src/dataset_comb/gen_dataset_partial_comb.py \
    >$HOME/tmp/NSC/codes/src/dataset_comb/gen_dataset_partial_comb.log 2>&1
