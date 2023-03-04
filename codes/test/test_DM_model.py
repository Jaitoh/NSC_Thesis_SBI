debug = False # debug set to True to compare python and matlab results

# append sys.path
import sys
sys.path.append('./src')

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parse_data.decode_parameter import decode_mat_fitted_parameters
from simulator.DM_model import DM_model
from parse_data.parse_trial_data import parse_trial_data

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

cmaps = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']
fig_dir = Path('../figures/model/comparison/Python')

# load .mat file
filePath = Path('../data/params/263 models fitPars/data_fitPars_S1.mat')
paramsFitted = decode_mat_fitted_parameters(filePath)
filePath = Path('../data/trials.mat')
trial_data = parse_trial_data(filePath)

pulses = trial_data['pulse']
subjectIDs = trial_data['subjID']==1
subjectIDs = subjectIDs.reshape(-1)

inputs = pulses[subjectIDs==1,:]
inputs = inputs[:,:]
# inputs = inputs[0:100,:]
input_len = inputs.shape[0]

num_models = len(paramsFitted['allModelsList'])
    
# probRs = -1*np.ones((num_models, input_len))

def run_model(idx):
    params = {}
    params['bias']   = paramsFitted['bias'][idx]
    params['sigmas'] = paramsFitted['sigmas'][idx,:]
    params['BGLS']   = paramsFitted['BGLS'][idx, :, :]
    params['modelName'] = paramsFitted['allModelsList'][idx]
    # print('Model: ' + paramsFitted['allModelsList'][idx])

    model = DM_model(params=params)

    probR_temp = -1*np.ones((1, input_len))

    for i in range(input_len):
        _, probR = model.stoch_simulation(inputs[i,:], debug=debug)
        probR_temp[0,i] = probR
        
    return probR_temp

if __name__ == '__main__':
    # with Pool(cpu_count()) as p:
    #     results = list(tqdm(p.imap(run_model, range(num_models)), total=num_models))
    probRs = -1*np.ones((num_models, input_len))
    for i in tqdm(range(num_models)):
        probR_temp = run_model(i)
        probRs[i,:] = probR_temp
    
    # save probRs
    np.save('../data/compare_python_matlab/probRs_python.npy', probRs)
