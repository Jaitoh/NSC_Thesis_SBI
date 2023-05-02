import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from dataset.data_process import process_x_seqC_part

class MyDataset(Dataset):
    """MyDataset class

    Returns:
        seqC :             -> (DMS, 15 or L_x)
        theta: (5000, 4)   -> (4,)
        probR: (DMS, 5000) -> (DMS, )
        
    """
    def __init__(self, 
                 data_path, 
                 
                 num_sets=None,
                 num_theta_each_set=5000, 
                 
                 seqC_process='norm',
                 nan2num=-1,
                 summary_type=0,
                 ):
        
        self.data_path = data_path

        # with h5py.File(self.data_path, 'r') as f:
        self.f = h5py.File(self.data_path, 'r', libver='latest', swmr=True)
            
        self.total_sets = len(self.f.keys()) if num_sets is None else num_sets
        self.total_samples = num_theta_each_set*self.total_sets
        self.num_theta_each_set = num_theta_each_set
        
        self.seqC_process = seqC_process
        self.nan2num = nan2num
        self.summary_type = summary_type

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        set_idx, theta_idx = divmod(idx, self.num_theta_each_set)

        # Load seqC, theta, and probR for the given idx
        f = self.f
        
        if self.seqC_process == 'norm':
            seqC = f[f'set_{set_idx}']['seqC_normed'][:]
        elif self.seqC_process == 'summary':
            if self.summary_type == 0:
                seqC = f[f'set_{set_idx}']['seqC_summary_0'][:]
            elif self.summary_type == 1:
                seqC = f[f'set_{set_idx}']['seqC_summary_1'][:]
        else:
            raise ValueError(f"seqC_process {self.seqC_process} not supported")
        # print(f"seqC shape: {seqC.shape}")

        theta = f[f'set_{set_idx}']['theta'][theta_idx,:]
        probR = f[f'set_{set_idx}']['probR'][:, theta_idx][:, np.newaxis]

        # print(f"theta shape: {theta.shape}")

        return torch.from_numpy(seqC), torch.from_numpy(theta), torch.from_numpy(probR)

class My_Dataset_Mem(Dataset):
    
    """My_Dataset_Mem:  load data into memory

    Returns:
        __getitem__:
        seqC :             -> (DMS, 15 or L_x)
        theta: (5000, 4)   -> (4,)
        probR: (DMS, 5000) -> (DMS, )
        
        all data:
        seqC_all:          (num_chosen_sets, DMS, 15 or L_x)
        theta_all:         (num_chosen_sets, num_chosen_theta_each_set, 4)
        probR_all:         (num_chosen_sets, DMS, num_chosen_theta_each_set)
        
    """
    def __init__(self, data_path, config, chosen_dur=[3,5,7,9,11,13,15]):
        
        num_chosen_sets = config['dataset']['num_chosen_sets']
        num_chosen_theta_each_set = config['dataset']['num_chosen_theta_each_set']
        
        self.data_path = data_path

        # with h5py.File(self.data_path, 'r') as f:
        f = h5py.File(self.data_path, 'r', libver='latest', swmr=True)

        self.total_sets             = len(f.keys())
        self.all_set_names          = list(f.keys())
        self.total_theta_in_a_set   = f[self.all_set_names[0]]['theta'].shape[0]
        
        num_chosen_sets = self.total_sets if num_chosen_sets is None else num_chosen_sets
        self.total_samples          = num_chosen_theta_each_set*num_chosen_sets
        self.num_chosen_theta_each_set  = num_chosen_theta_each_set

        seqC_process    = config['dataset']['seqC_process']
        nan2num         = config['dataset']['nan2num']
        summary_type    = config['dataset']['summary_type']
        DMS             = f[self.all_set_names[0]]['seqC_normed'].shape[0]
        D, M, S         = f[self.all_set_names[0]]['seqC'].shape[:-1]
        
        
        # Randomly choose sets and theta
        randomly_chosen_set_idx   = np.random.choice(self.all_set_names, num_chosen_sets, replace=False)
        randomly_chosen_theta_idx = np.random.choice(self.total_theta_in_a_set, num_chosen_theta_each_set, replace=False)
        
        # Load data into memory seqC_all, theta_all, probR_all
        seqC_all  = np.empty((num_chosen_sets, DMS, 15))
        theta_all = np.empty((num_chosen_sets, num_chosen_theta_each_set, 4))
        probR_all = np.empty((num_chosen_sets, DMS, num_chosen_theta_each_set))
        
        for set_idx, set_name in enumerate(randomly_chosen_set_idx):
            if seqC_process == 'norm':
                seqC_all[set_idx] = f[set_name]['seqC_normed'][:]
            elif seqC_process == 'summary':
                if summary_type == 0:
                    seqC_all[set_idx] = f[set_name]['seqC_summary_0'][:]
                elif summary_type == 1:
                    seqC_all[set_idx] = f[set_name]['seqC_summary_1'][:]
            else:
                raise ValueError(f"seqC_process {seqC_process} not supported")

            theta_all[set_idx] = f[set_name]['theta'][:][randomly_chosen_theta_idx, :]
            probR_all[set_idx] = f[set_name]['probR'][:][:, randomly_chosen_theta_idx]
        
        # set the values that are not chosen in DMS to 0
        idx_not_chosen = self._get_idx_not_chosen(chosen_dur, D, M, S)
        
        seqC_all[:, idx_not_chosen, :] = 0
        probR_all[:, idx_not_chosen, :] = 0
        
        self.seqC_all = seqC_all
        self.theta_all = theta_all
        self.probR_all = probR_all
        
        print(f"dur of {list(chosen_dur)} are chosen, others are set to 0")
        print(f"current dataset seqC of shape: {self.seqC_all.shape}")
        print(f"current dataset theta of shape: {self.theta_all.shape}")
        print(f"current dataset probR of shape: {self.probR_all.shape}")

        f.close()
    
    
    def _get_idx_not_chosen(self, chosen_dur, D, M, S):
        """ return the idx where the corresponding value is not chosen by the given chosen_idx
        dur_list start from 3, end at 15 with step 2
        """
        chosen_dur = np.array(chosen_dur)
        starting_idxs, ending_idxs = ((chosen_dur-3)/2)*M*S, ((chosen_dur-3)/2+1)*M*S
        
        idx_not_chosen = np.array(range(int(D*M*S)))
        for start, end in zip(starting_idxs, ending_idxs):
            idx_not_chosen = np.setdiff1d(idx_not_chosen, range(int(start), int(end)))

        return idx_not_chosen

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        set_idx, theta_idx = divmod(idx, self.num_chosen_theta_each_set)

        seqC  = self.seqC_all[set_idx]
        theta = self.theta_all[set_idx, theta_idx]
        probR = self.probR_all[set_idx, :, theta_idx][:, np.newaxis]
        
        return torch.from_numpy(seqC), torch.from_numpy(theta), torch.from_numpy(probR)
   
class Data_Prefetcher():
    
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # self.next_input = self.next_input.float()
        
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_ = self.next_input
        target = self.next_target
        self.preload()
        return input_, target
            
    
def collate_fn(batch, config, debug=False):
    
    C = config['dataset']['num_probR_sample']
    
    if debug:
        start_time_0 = time.time()
    
    x_batch, theta_batch = [], []
    
    x_batch     = torch.empty((C * len(batch), batch[0][0].shape[0], batch[0][0].shape[1]+batch[0][2].shape[1]))
    theta_batch = torch.empty((C * len(batch), batch[0][1].shape[0]))
    
    for i, (seqC, theta, probR) in enumerate(batch): # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
        
        probR     = probR.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, D*M*S, 1)
        x_seqC    = seqC.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, D*M*S, 15)
        x_choice  = torch.bernoulli(probR) # (C, D*M*S, 1)
        
        x         = torch.cat([x_seqC, x_choice], dim=-1)
        theta = theta.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, 4)
        
        x_batch[i*C:(i+1)*C] = x
        theta_batch[i*C:(i+1)*C] = theta
    
    if debug:
        print(f"\ncollate_fn: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")
    
    if debug:
        start_time = time.time()
    # Shuffle x along the 2nd axis
    x_batch = torch.stack([x_batch[i][torch.randperm(x_batch.shape[1])] for i in range(x_batch.shape[0])])
    if debug:
        print(f"collate_fn: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
    
    if debug:
        start_time = time.time()
    # Shuffle the batched dataset
    indices     = torch.randperm(x_batch.shape[0])
    x_batch     = x_batch[indices]
    theta_batch = theta_batch[indices]
    if debug:
        print(f"collate_fn: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        print(f"collate_fn: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")
    
    return x_batch.to(torch.float32), theta_batch.to(torch.float32)


def collate_fn_vec(batch, config, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
    """
    
    C = config['dataset']['num_probR_sample']
    B = len(batch)
    
    if debug:
        start_time_0 = time.time()
    
    seqC_shape = batch[0][0].shape
    theta_shape = batch[0][1].shape
    probR_shape = batch[0][2].shape
    
    # Preallocate tensors
    seqC_batch = torch.empty((B, *seqC_shape))
    theta_batch = torch.empty((B, *theta_shape))
    probR_batch = torch.empty((B, *probR_shape))
    
    # Fill tensors with data from the batch
    for i, (seqC, theta, probR) in enumerate(batch):
        seqC_batch[i] = seqC
        theta_batch[i] = theta
        probR_batch[i] = probR
    if debug:
        print(f"collate_fn_vec: dataloading {(time.time() - start_time_0)*1000:.2f} ms")
    
    # Repeat seqC and theta C times along a new dimension
    x_seqC = seqC_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 15) first C samples are the same, from the first batch
    theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)
    
    # Repeat probR C times along a new dimension and sample from Bernoulli distribution
    probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
    x_choice = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)
    
    # Concatenate x_seqC and x_choice
    x_batch = torch.cat([x_seqC, x_choice], dim=-1)  # (C*B, D*M*S, 16)
    if debug:
        print(f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")
    
    if debug:
        start_time = time.time()
    # Shuffle x along the 2nd axis
    # x_batch = torch.stack([x_batch[i,:,:][torch.randperm(x_batch.shape[1]),:] for i in range(x_batch.shape[0])])
    DMS = x_batch.shape[1]
    x_batch_shuffled = torch.empty_like(x_batch)
    for i in range(B*C):
        x_batch_shuffled[i] = x_batch[i][torch.randperm(DMS)]
    
    if debug:
        print(f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
    
    if debug:
        start_time = time.time()
    
    # Shuffle the batched dataset
    indices             = torch.randperm(x_batch_shuffled.shape[0])
    x_batch_shuffled    = x_batch_shuffled[indices]
    theta_batch         = theta_batch[indices]
    if debug:
        print(f"collate_fn_vec: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        print(f"collate_fn_vec: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")
    
    return x_batch_shuffled.to(torch.float32), theta_batch.to(torch.float32)


def collate_fn_probR(batch, Rchoice_method='probR_sampling', num_probR_sample=10):
    ''' OLD VERSION
    batch is a list of tuples, each tuple is (theta, x, prior_masks) 
        original shapes:
        theta.shape = (T*C, L_theta)
        x.shape     = (T*C, DMS, L_x)
        (sequence should be shuffled when batched into the dataloader)
        
        e.g. C = 1  if Rchoice_method == 'probR'
        e.g. C = 10 if Rchoice_method == 'probR_sampling'
    ''' 
    
    theta, x, _ = zip(*batch)
    
    theta   = torch.stack(theta)
    x       = torch.stack(x)
    _       = torch.stack(_)
    
    if Rchoice_method == 'probR':
        # repeat theta and x for each probR sample
        theta_new   = theta.repeat_interleave(num_probR_sample, dim=0) # (T*C, L_theta)
        x_new       = x.repeat_interleave(num_probR_sample, dim=0) # (T*C, DMS, 15+1)
        _           = _.repeat_interleave(num_probR_sample, dim=0) # (T*C, 1)
        x_seqC      = x_new[:, :, :-1] # (T*C, DMS, 15)
        x_probRs    = x_new[:, :, -1].unsqueeze_(dim=2) # (T*C, DMS, 1)
        
        # sample Rchoice from probR with Bernoulli
        x_Rchoice   = torch.bernoulli(x_probRs)
        x           = torch.cat((x_seqC, x_Rchoice), dim=2)
        theta       = theta_new
        
    return theta, x, _


if __name__ == '__main__':
    batch_size = 20
    num_probR_sample = 100

    # Configuration
    config = {
        'dataset': {
            'num_probR_sample'  : num_probR_sample,
            'num_chosen_sets'       : 9,
            'num_chosen_theta_each_set' : 500, #500
            'seqC_process'      : 'norm',
            'nan2num'           : -1,
            'summary_type'      : 0
        }
    }

    # dataset_path = '../../../data/dataset/dataset_L0_exp_set_0_test.h5'
    dataset_path = '../data/dataset/dataset_L0_exp_set_0_test.h5'
    
    debug = True
    dataset = MyDataset(dataset_path)
    # dataset = My_Dataset_Mem(dataset_path, config)
    counter = 0
    
    case = 1
    
    if case == 0:
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                pin_memory=True, num_workers=2,
                                collate_fn=lambda b: collate_fn(b, config, debug))
    elif case == 1: # vec version
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                pin_memory=True, num_workers=2,
                                collate_fn=lambda b: collate_fn_vec(b, config, debug))
    elif case == 2: # prefected version
        data_loader = DataLoader(dataset, batch_size=batch_size, 
                                pin_memory=True, num_workers=2,
                                collate_fn=lambda b: collate_fn_vec(b, config, debug))
        prefetcher = Data_Prefetcher(data_loader)
    
    elif case == 3: # load to memory
        print()
        
    
    if debug:
        start_time = time.time()
    
    if case == 2:
        data, label = prefetcher.next()
        iteration = 1
        while data is not None:
            data, label = prefetcher.next()
            iteration += 1
            if iteration == 5:
                break
    else:
        for x_batch, theta_batch in data_loader:
            counter += 1
            if counter == 5:
                counter = 0
                break
            
    
    print(f"--- finished {case} in: {(time.time() - start_time)*1000:.2f} ms")