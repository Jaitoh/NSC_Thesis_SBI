import torch
import sbi
from sbi import utils as utils
from sbi.inference import infer
import sys
sys.path.append('./src')
print(sys.path)
from inference.infer import my_infer


# define the prior range  
subjID      = 1 # subject ID ranging from 1-15
modelName   = 'B-G-L0S-O-N-'
prior_min   = [-2,  0, 0,  0, -10]
prior_max   = [ 2, 10, 0, 10,  10]
prior       = utils.torchutils.BoxUniform(
                    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

# prepare data for training

posterior = my_infer( prior=prior,
                     method='SNPE',
                     num_sample=100,
                     modelName=modelName,
                     subjID=subjID,
                     trial_data_dir='../data/trials.mat',
                     save_data_dir='../data/data_for_sbi_sample100_s1.h5',
                     num_workers=12,
                     )