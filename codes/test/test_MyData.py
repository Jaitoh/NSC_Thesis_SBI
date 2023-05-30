import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gc
# from dataset.data_process import process_x_seqC_part
import sys
sys.path.append('./src')
from dataset.data_process import process_x_seqC_part

def generate_permutations(N, K):
    """
    Generate random permutations.

    Args:
        N (int): The number of permutations to generate.
        K (int): The length of each permutation.

    Returns:
        torch.Tensor: A tensor of shape (N, K) containing random permutations.

    """
    permutations = torch.rand(N, K).argsort(dim=-1)
    return permutations

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def apply_advanced_indexing_along_dim1(tensor, indices):
    idx0 = torch.arange(tensor.size(0))[:, None].expand(indices.size(0), indices.size(1))
    return tensor[idx0, indices]

def get_L_seqC(seqC_process, summary_type):
    if seqC_process == 'norm':
        # seqC_shape = f[chosen_set_names[0]]['seqC_normed'].shape[1]
        L = 15
    elif seqC_process == 'summary':
        if summary_type == 0:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_0'].shape[1]
            L = 11
        elif summary_type == 1:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_1'].shape[1]
            L = 8
    return L
    
class My_HighD_Sets(Dataset):
    def __init__(self, data_path, config,
                 chosen_set_names, num_chosen_theta_each_set,
                 chosen_dur=[3,9,15], crop_dur=True, ):
        super().__init__()
        
        """Loading the high-dimensional sets into memory
        seqC            of shape set_0:(D, M, S, 15)  - (7, 3, 700, 15)
        seqC_normed     of shape set_0:(DMS, 15)      - (14700, 15) #TODO add this part into the dataset (currently removed)
        seqC_summary_0  of shape set_0:(DMS, 15)      - (14700, 11)
        seqC_summary_1  of shape set_0:(DMS, 15)      - (14700, 8)
        
        probR           of shape set_0:(D, M, S, T)   - (7, 3, 700, 5000)
        theta           of shape set_0:(T, 4)         - (5000, 4)
        
        loaded into memory:
        seqC_normed     of shape (num_sets, DM, S, L) - (7, 21, 700, 15)
        theta           of shape (num_sets, T, 4)     - (7, 5000, 4)
        probR           of shape (num_sets, DM, S, T) - (7, 21, 700, 5000) #TODO fixed probR sampling, in init function
        
        get item:
        seqC_normed     of shape (DM, S, L)           - (21, 700, 15)
        theta           of shape (4)                  - (4)
        probR           of shape (DM, S, 1)           - (21, 700, 1)
        """
        dur_list = [3,5,7,9,11,13,15]
        print(f'Loading {len(chosen_set_names)} dataset into memory...', end=' ')
        
        start_loading_time = time.time()
        self.num_chosen_theta_each_set = num_chosen_theta_each_set
        self.chosen_set_names = chosen_set_names
        
        f = h5py.File(data_path, 'r', libver='latest', swmr=True)
        max_theta_in_a_set = config['dataset']['num_max_theta_each_set']
        self.total_samples = num_chosen_theta_each_set*len(chosen_set_names)
        
        # load configurations
        dataset_config = config['dataset']
        seqC_process, nan2num, summary_type = dataset_config['seqC_process'], dataset_config['nan2num'], dataset_config['summary_type']
        L_seqC = get_L_seqC(seqC_process, summary_type)
        D, M, S, DMS = (*f[chosen_set_names[0]]['seqC'].shape[:-1], np.prod(f[chosen_set_names[0]]['seqC'].shape[:-1]))
        
        # define the final shape of the data
        randomly_chosen_theta_idx = np.random.choice(max_theta_in_a_set, num_chosen_theta_each_set, replace=False)
        chosen_dur_idx  = [dur_list.index(dur) for dur in chosen_dur] # mapping from [3, 5, 7, 9, 11, 13, 15] to [0, 1, 2, 3, 4, 5, 6]
        chosen_D = len(chosen_dur) if crop_dur else D
        self.seqC_all   = np.empty((len(chosen_set_names), chosen_D, M, S, L_seqC))         # (n_set, D, M, S, L)
        self.theta_all  = np.empty((len(chosen_set_names), num_chosen_theta_each_set, 4))   # (n_set, T, 4)
        self.probR_all  = np.empty((len(chosen_set_names), chosen_D, M, S, num_chosen_theta_each_set, 1))  # (n_set, D, M, S, T, 1)
        
        counter = 0
        for set_idx, set_name in enumerate(chosen_set_names):
            
            if counter % 10 == 0:
                print(counter, end=' ')
                
            seqC_data = self._get_seqC_data(crop_dur, f, seqC_process, summary_type, chosen_dur_idx, set_name) # (D, M, S, L)
            probR_data = f[set_name]['probR'][chosen_dur_idx, :, :, :][:, :, :, randomly_chosen_theta_idx] if crop_dur else f[set_name]['probR'][:, :, :, randomly_chosen_theta_idx] # (D, M, S, T)
            self.theta_all[set_idx] = f[set_name]['theta'][:][randomly_chosen_theta_idx, :] # (T, 4)
            self.seqC_all[set_idx] = seqC_data
            self.probR_all[set_idx] = probR_data
            del seqC_data, probR_data
            counter += 1

        if not crop_dur:
            self.seqC_all [:, chosen_dur_idx, :, :, :]  = 0 # (n_set, D, M, S, L)
            self.probR_all[:, chosen_dur_idx, :, :, :]  = 0 # (n_set, D, M, S, T, 1)
        
        self._print_info(chosen_dur, crop_dur, start_loading_time)
        
        f.close()
        
        # convert to tensor and reshape
        self.seqC_all  = torch.from_numpy(self.seqC_all).reshape(len(chosen_set_names), chosen_D*M, S, L_seqC).to(torch.float32).contiguous() # (n_set, DM, S, L)
        self.probR_all = torch.from_numpy(self.probR_all).reshape(len(chosen_set_names), chosen_D*M, S, num_chosen_theta_each_set).to(torch.float32).contiguous() # (n_set, DM, S, T)
        self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()
        

    def _print_info(self, chosen_dur, crop_dur, start_loading_time):
        
        print(f" finished in: {time.time()-start_loading_time:.2f}s")
        if crop_dur:
            print(f"dur of {list(chosen_dur)} are chosen, others are [removed] ")
        else:
            print(f"dur of {list(chosen_dur)} are chosen, others are [set to 0] (crop_dur is suggested to be set as True)")
        print(f"[seqC] shape: {self.seqC_all.shape}")
        print(f"[theta] shape: {self.theta_all.shape}")
        print(f"[probR] shape: {self.probR_all.shape}")
        
        
    def _get_seqC_data(self, crop_dur, f, seqC_process, summary_type, chosen_dur_idx, set_name):
        
        seqC = f[set_name]['seqC'][chosen_dur_idx, :, :, :] if crop_dur else f[set_name]['seqC'][:] # (D, M, S, L)
        seqC_data = process_x_seqC_part(
                seqC                = seqC, 
                seqC_process        = seqC_process, # 'norm' or 'summary'
                nan2num             = -1,
                summary_type        = summary_type, # 0 or 1
            )
        
        return seqC_data # (D, M, S, L)
            
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        
        """ #TODO probR sampling in init function
        get a sample from the dataset
        seqC    (DM, S, L)
        theta   (4)
        probR   (DM, S, 1)
        """
        
        set_idx, theta_idx = divmod(idx, self.num_chosen_theta_each_set) # faset than unravel_index indexing
        
        # set_idx, theta_idx = self.set_idxs[idx], self.theta_idxs[idx]

        seqC  = self.seqC_all[set_idx]
        theta = self.theta_all[set_idx, theta_idx]
        probR = self.probR_all[set_idx, :, :, theta_idx][:, :, np.newaxis]
        
        return seqC, theta, probR

import sys
sys.path.append('./src')
from config.load_config import load_config

data_path = "../data/dataset/dataset_L0_exp_set_0.h5"
CONFIG_SIMULATOR_PATH = "./src/config/simulator/exp_set_0.yaml"
CONFIG_DATASET_PATH = "./src/config/dataset/dataset-p2-1.yaml"
CONFIG_TRAIN_PATH="./src/config/train/train-p2-2.yaml"

config = load_config(config_simulator_path=CONFIG_SIMULATOR_PATH,
                    config_dataset_path=CONFIG_DATASET_PATH,
                    config_train_path=CONFIG_TRAIN_PATH,)
    
dataset = My_HighD_Sets(data_path, config,
              chosen_set_names=['set_0', 'set_99'], num_chosen_theta_each_set=500,
              chosen_dur=[3,9,15], crop_dur=True)

print(len(dataset))
print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)