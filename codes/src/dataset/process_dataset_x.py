# process the seqC part of the input x and store the results
import argparse
import h5py
from tqdm import tqdm

import os
print(os.getcwd())
import sys
sys.path.append('./src')
print(sys.path)

# from dataset.data_process import process_x_seqC_part
from data_process import process_x_seqC_part

args = argparse.ArgumentParser()
args.add_argument('--data_path', type=str, default='../data/dataset/dataset_L0_exp_set_0.h5')
args = args.parse_args()

data_path = args.data_path

# process and write the dataset
with h5py.File(data_path, 'r+') as f:

    set_list = list(f.keys())
    print("preprocessing the seqC ...")
    
    for one_set in tqdm(set_list):
            
        seqC = f[one_set]['seqC']

        seqC_normed = process_x_seqC_part(
            seqC                = seqC, 
            seqC_process        = 'norm',
            nan2num             = -1,
            summary_type        = 0,
        )
        the_shape = seqC_normed.shape
        seqC_normed = seqC_normed.reshape(the_shape[0]*the_shape[1]*the_shape[2], the_shape[3])
        
        seqC_summary_0 = process_x_seqC_part(
            seqC                = seqC, 
            seqC_process        = 'summary',
            nan2num             = -1,
            summary_type        = 0,
        )
        
        the_shape = seqC_summary_0.shape
        seqC_summary_0 = seqC_summary_0.reshape(the_shape[0]*the_shape[1]*the_shape[2], the_shape[3])

        seqC_summary_1 = process_x_seqC_part(
            seqC                = seqC, 
            seqC_process        = 'summary',
            nan2num             = -1,
            summary_type        = 1,
        )

        the_shape = seqC_summary_1.shape
        seqC_summary_1 = seqC_summary_1.reshape(the_shape[0]*the_shape[1]*the_shape[2], the_shape[3])

        if 'seqC_normed' in f[one_set].keys(): #update the dataset
            f[one_set].pop('seqC_normed')
            f[one_set].pop('seqC_summary_0')
            f[one_set].pop('seqC_summary_1')
        
        f[one_set].create_dataset('seqC_normed', data=seqC_normed)
        f[one_set].create_dataset('seqC_summary_0', data=seqC_summary_0)
        f[one_set].create_dataset('seqC_summary_1', data=seqC_summary_1)
        
# TODO profiling the collate_fn in this file