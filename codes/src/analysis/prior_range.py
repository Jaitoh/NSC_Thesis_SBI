import pickle
import numpy as np
import argparse
from tqdm import tqdm
import sys
sys.path.append('./src')

from codes.src.simulator.model_sim_pR import DM_sim_for_seqCs_parallel as sim_parallel
from codes.src.simulator.model_sim_pR import get_boxUni_prior
from utils.set_seed import setup_seed

setup_seed(100)

# from dataset.seqC_generator import seqC_combinatorial_generator

# output_seqs = {}

# dur_list = range(3, 15+1, 2)
# ms_list = [0.2, 0.4, 0.8]

# for dur in tqdm(dur_list):
#     seq = seqC_combinatorial_generator(dur)
#     output_seqs[f'dur_{dur}'] = seq

# save the output
# import pickle
# with open('../data/seqC_combinatorial.pkl', 'wb') as f:
#     pickle.dump(output_seqs, f)
# print('file saved to ../data/seqC_combinatorial.pkl')

# get input arguments
args = argparse.ArgumentParser()
args.add_argument('--dur_list', type=str, default='[3,5,7,9,11,13,15]')
args.add_argument('--task_part', type=str, default='[0, 1]')
args = args.parse_args()

# dur_list = range(3, 15+1, 2)
dur_list = eval(args.dur_list)
task_part = eval(args.task_part)
print(f'dur_list: {dur_list}')
print(f'task_part: {task_part}')
ms_list = [0.2, 0.4, 0.8]

with open('../data/seqC_combinatorial.pkl', 'rb') as f:
    output_seqs = pickle.load(f)
    
for dur in dur_list:
    print(f'dur_{dur:2} number of possible combinations: {len(output_seqs[f"dur_{dur:}"]):7}')

# prior
num_prior_sample = 500
prior_min = [-2.5,   0,  0, -11]
prior_max = [ 2.5,  77, 18,  10]

prior = get_boxUni_prior(prior_min, prior_max)
params = prior.sample((num_prior_sample,)).cpu().numpy()
# print first 5 and last 5 params
print(f'first 5 params:\n {params[:5]}')
print(f'last 5 params:\n {params[-5:]}')

# reshape for parallel probR computation
for dur in dur_list:
    print(f"\n===\ncalculating probR for each seqC of dur_{dur}...")
    seqs = np.empty((1,3,*output_seqs[f"dur_{dur}"].shape))
    for i, ms in enumerate(ms_list):
        seqs[:,i,:] = output_seqs[f"dur_{dur}"]*ms_list[i]

    num_parts = 100 if dur > 8 else 1
    # subdivide seqs into 16 chunks along the 3rd axis
    seqs_divided = np.array_split(seqs, num_parts, axis=2)
    print(f'reshaped seqs of size: {seqs.shape}\nsubdivide into {len(seqs_divided)} chunks along the 3rd axis\nthe divided seqs has size: {seqs_divided[0].shape}')

    
    # theta_collection = []
    # probR_collection = []
    
    task_nums = np.arange(task_part[0], task_part[1])
    for i, seq in enumerate(seqs_divided[task_part[0]:task_part[1]]):
        print(f'chunk {task_nums[i]} of {task_part[0]} to {task_part[1]}')
        seq, theta, probR = sim_parallel(
            seqCs = seq,
            prior = params,
            num_prior_sample = num_prior_sample,
            model_name='B-G-L0S-O-N-',
            num_workers=16,
            privided_prior=True,
        )
        
        # theta_collection.append(theta)
        # probR_collection.append(probR)

        # save the output
        with open(f'/home/wehe/scratch/data/prior_sim/seqC_combinatorial_theta_dict_dur_{dur}_part{task_nums[i]}.pkl', 'wb') as f:
            pickle.dump(theta, f)
        with open(f'/home/wehe/scratch/data/prior_sim/seqC_combinatorial_probR_dict_dur_{dur}_part{task_nums[i]}.pkl', 'wb') as f:
            pickle.dump(probR, f)
