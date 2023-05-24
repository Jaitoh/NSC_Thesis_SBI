import time
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import gc
# from dataset.data_process import process_x_seqC_part

def generate_permutations(N, K):
    """
    Generate random permutations.

    Args:
        N (int): The number of permutations to generate.
        K (int): The length of each permutation.

    Returns:
        torch.Tensor: A tensor of shape (N, K) containing random permutations.

    """
    # time_ = time.time()
    # print(f'generate permutations of size {N, K}...', end = ' ')
    
    # Generate random values between 0 and 1
    # Sort the random values along the last dimension to obtain permutations
    permutations = torch.rand(N, K).argsort(dim=-1)
    # faster than:  permutations = torch.stack([torch.randperm(K) for _ in range(N)], dim=0)
    # print(f'in {(time.time() - time_)/60:.2f} min, MEM size {permutations.element_size() * permutations.nelement() / 1024**3:.2f} GB\n')
    
    return permutations

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

# def apply_advanced_indexing_along_dim1(tensor, indices):
#     idx0 = torch.arange(tensor.size(0))[:, None].expand_as(indices)
#     idx2 = torch.arange(tensor.size(2))[None, :].expand_as(indices)
#     return tensor[idx0, indices, idx2]

def apply_advanced_indexing_along_dim1(tensor, indices):
    idx0 = torch.arange(tensor.size(0))[:, None].expand(indices.size(0), indices.size(1))
    return tensor[idx0, indices]

class My_Chosen_Sets(Dataset):
    def __init__(self, data_path, config, chosen_set_names, num_chosen_theta_each_set, chosen_dur=[3,5,7,9,11,13,15], crop_dur=False):
        """Loading num_chosen_theta_each_set 
        from the chosen sets into memory
        from data_path
        """
        
        print(f'Loading {len(chosen_set_names)} dataset into memory...', end=' ')
        start_loading_time = time.time()
        self.num_chosen_theta_each_set = num_chosen_theta_each_set
        self.chosen_set_names = chosen_set_names
        
        f = h5py.File(data_path, 'r', libver='latest', swmr=True)
        # total_theta_in_a_set = f[chosen_set_names[0]]['theta'].shape[0]
        total_theta_in_a_set = config['dataset']['num_max_theta_each_set']
        self.total_samples = num_chosen_theta_each_set*len(chosen_set_names)
        
        seqC_process    = config['dataset']['seqC_process']
        nan2num         = config['dataset']['nan2num']
        summary_type    = config['dataset']['summary_type']
        DMS             = f[chosen_set_names[0]]['seqC_normed'].shape[0]
        D, M, S         = f[chosen_set_names[0]]['seqC'].shape[:-1]
        
        randomly_chosen_theta_idx = np.random.choice(total_theta_in_a_set, num_chosen_theta_each_set, replace=False)
        
        if seqC_process == 'norm':
            seqC_shape = f[chosen_set_names[0]]['seqC_normed'].shape[1]
        elif seqC_process == 'summary':
            if summary_type == 0:
                seqC_shape = f[chosen_set_names[0]]['seqC_summary_0'].shape[1]
            elif summary_type == 1:
                seqC_shape = f[chosen_set_names[0]]['seqC_summary_1'].shape[1]
        
        # set the values that are not chosen in DMS to 0, or remove them
        DMS_idx_not_chosen, DMS_idx_chosen = self._get_idx_not_chosen(chosen_dur, D, M, S)
        num_chosen_DMS = len(DMS_idx_chosen) if crop_dur else DMS
        self.seqC_all  = np.empty((len(chosen_set_names), num_chosen_DMS, seqC_shape), dtype=np.float32) # shape (num_chosen_sets, DMS, seqC_shape)
        self.theta_all = np.empty((len(chosen_set_names), num_chosen_theta_each_set, 4), dtype=np.float32) # shape (num_chosen_sets, num_chosen_theta_each_set, 4)
        self.probR_all = np.empty((len(chosen_set_names), num_chosen_DMS, num_chosen_theta_each_set), dtype=np.float32) # shape (num_chosen_sets, DMS, num_chosen_theta_each_set)
        
        counter = 0
        for set_idx, set_name in enumerate(chosen_set_names):
            if counter % 10 == 0:
                print(counter, end=' ')
            seqC_data = self._get_seqC_data(crop_dur, f, seqC_process, summary_type, DMS_idx_chosen, set_name)
            probR_data = f[set_name]['probR'][DMS_idx_chosen, :][:, randomly_chosen_theta_idx] if crop_dur else f[set_name]['probR'][:, randomly_chosen_theta_idx]
            
            self.theta_all[set_idx] = f[set_name]['theta'][:][randomly_chosen_theta_idx, :]
            self.seqC_all[set_idx]  = seqC_data
            self.probR_all[set_idx] = probR_data
            del seqC_data, probR_data
            counter += 1
            
        # set the values that are not chosen in DMS to 0, or remove them
        if not crop_dur: # set not chosen to 0
            self.seqC_all[:, DMS_idx_not_chosen, :] = 0
            self.probR_all[:, DMS_idx_not_chosen, :] = 0
            
        print(f" finished in: {time.time()-start_loading_time:.2f}s")
        if crop_dur:
            print(f"dur of {list(chosen_dur)} are chosen, others are [removed] ")
        else:
            print(f"dur of {list(chosen_dur)} are chosen, others are [set to 0] (crop_dur is suggested to be set as True)")
        print(f"[seqC] shape: {self.seqC_all.shape}")
        print(f"[theta] shape: {self.theta_all.shape}")
        print(f"[probR] shape: {self.probR_all.shape}")
        
        f.close()
        del f
        
        # convert to tensor
        self.seqC_all  = torch.from_numpy(self.seqC_all).to(torch.float32).contiguous()
        self.theta_all = torch.from_numpy(self.theta_all).to(torch.float32).contiguous()
        self.probR_all = torch.from_numpy(self.probR_all).to(torch.float32).contiguous()
        
    def _get_seqC_data(self, crop_dur, f, seqC_process, summary_type, DMS_idx_chosen, set_name):
        
        if seqC_process == 'norm':
            seqC_data = f[set_name]['seqC_normed'][DMS_idx_chosen, :] if crop_dur else f[set_name]['seqC_normed'][:]
        
        elif seqC_process == 'summary':
        
            if summary_type == 0:
                seqC_data = f[set_name]['seqC_summary_0'][DMS_idx_chosen, :] if crop_dur else f[set_name]['seqC_normed'][:]
        
            elif summary_type == 1:
                seqC_data = f[set_name]['seqC_summary_1'][DMS_idx_chosen, :] if crop_dur else f[set_name]['seqC_normed'][:]
        
        else:
            raise ValueError(f"seqC_process {seqC_process} not supported")
        
        return seqC_data
    
    def _get_idx_not_chosen(self, chosen_dur, D, M, S):
        """ return the idx where the corresponding value is not chosen by the given chosen_idx
        dur_list start from 3, end at 15 with step 2
        """
        chosen_dur = np.array(chosen_dur)
        starting_idxs, ending_idxs = ((chosen_dur-3)/2)*M*S, ((chosen_dur-3)/2+1)*M*S
        
        idx_chosen = np.array([], dtype=np.int32)
        idx_not_chosen = np.array(range(int(D*M*S)), dtype=np.int32)
        for start, end in zip(starting_idxs, ending_idxs):
            idx_not_chosen  = np.setdiff1d(idx_not_chosen, range(int(start), int(end)))
            idx_chosen      = np.append(idx_chosen, range(int(start), int(end)))
        
        return idx_not_chosen, idx_chosen

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        # tic = time.time()
        set_idx, theta_idx = divmod(idx, self.num_chosen_theta_each_set) # faset than unravel_index indexing
        
        # set_idx, theta_idx = self.set_idxs[idx], self.theta_idxs[idx]

        seqC  = self.seqC_all[set_idx]
        theta = self.theta_all[set_idx, theta_idx]
        probR = self.probR_all[set_idx, :, theta_idx][:, np.newaxis]
        
        # print(f"get item time: {(time.time()-tic)*1000:.2f} ms")
        return seqC, theta, probR
        # return torch.from_numpy(seqC), torch.from_numpy(theta), torch.from_numpy(probR)
        
class My_Processed_Dataset(My_Chosen_Sets):
    def __init__(self, data_path, config, chosen_set_names, num_chosen_theta_each_set, chosen_dur=[3,5,7,9,11,13,15], crop_dur=False):
        """My_Dataset:  load data into memory and finish the preprocessing before training

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
        
        super().__init__(data_path, config, chosen_set_names, num_chosen_theta_each_set, chosen_dur, crop_dur)
        # self.seqC_all = seqC_all  # shape (num_chosen_sets, D_MS, 15 or L_x)
        # self.theta_all = theta_all # shape (num_chosen_sets, num_chosen_theta_each_set, 4)
        # self.probR_all = probR_all # shape (num_chosen_sets, D_MS, num_chosen_theta_each_set)
        
        # Repeat probR C times along a new dimension and sample from Bernoulli distribution
        self.C = config['dataset']['num_probR_sample']
        print(f"\nSampling {self.C} times from probR ... ", end="")
        time_start = time.time()
        self.DMS = self.seqC_all.shape[1]
        self.probR_all = self.probR_all.repeat_interleave(self.C, dim=-1)  # (num_chosen_sets, D_*M*S, num_chosen_theta_each_set*C)
        self.probR_all = torch.bernoulli(self.probR_all).unsqueeze(-1).contiguous()  # (num_chosen_sets, D*M*S, num_chosen_theta_each_set*C)
        print(f"in {(time.time()-time_start)/60:.2f}min")
        
        self.total_samples = len(chosen_set_names) * self.num_chosen_theta_each_set * self.C
        print(f"sampled probR shape {self.probR_all.shape} MEM size {self.probR_all.element_size()*self.probR_all.nelement()/1024**3:.2f}GB, Total samples: {self.total_samples} ") 
        self.permutations = generate_permutations(self.total_samples, self.DMS).contiguous() # (CTSet, D_MS)
        
        self.seqC_all = self.seqC_all.contiguous()
        self.theta_all = self.theta_all
        
        indices = torch.arange(self.total_samples)
        self.set_idxs, self.theta_idxs, self.probR_sample_idxs = unravel_index(indices, (len(self.chosen_set_names), self.num_chosen_theta_each_set, self.C))
        
        
    def __len__(self):
        return self.total_samples

    
    def __getitem__(self, idx):
        
        # time_start = time.time()
        # Calculate set index and theta index within the set
        permutation = self.permutations[idx]
        set_idx, theta_idx, probR_sample_idx = self.set_idxs[idx], self.theta_idxs[idx], self.probR_sample_idxs[idx]

        x = torch.empty((self.DMS, self.seqC_all.shape[2]+1), dtype=torch.float32)
        x[:, :self.seqC_all.shape[2]] = self.seqC_all[set_idx, permutation, :] # (D_MS, 15 or L_x)
        # print(f"getitem seqC: {(time.time()-time_start)*1000:.2f}ms")
        # time_start = time.time()
        x[:, self.seqC_all.shape[2]:] = self.probR_all[set_idx, permutation, theta_idx * self.C + probR_sample_idx] # (D_MS, 1)
        # print(f"getitem probR_all: {(time.time()-time_start)*1000:.2f}ms")
        
        theta = self.theta_all[set_idx, theta_idx] # (4,)
        
        return x, theta
    
class Data_Prefetcher():
    
    def __init__(self, loader, prefetch_factor=3):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.prefetch_factor = prefetch_factor
        self.prefetched_data = []
        self.preload()

    def __len__(self):
        return len(self.loader)
    
    def preload(self):
        try:
            for _ in range(self.prefetch_factor):
                input, target = next(self.loader)
                with torch.cuda.stream(self.stream):
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                self.prefetched_data.append((input, target))
                # print(f'prefetcher preloaded {len(self.prefetched_data)}')
                
        except StopIteration:
            self.prefetched_data.append((None, None))

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if len(self.prefetched_data) == 0:
            return None, None

        input_, target = self.prefetched_data.pop(0)
        # print(f'prefetcher next called: prefetcher {len(self.prefetched_data)}')
        self.preload()  # start preloading next batches
        return input_, target
    
        # torch.cuda.current_stream().wait_stream(self.stream)
        # input_ = self.next_input
        # target = self.next_target
        # self.preload()
        # return input_, target

def collate_fn_vec(batch, config, shuffling_method=0, debug=False):
    """
    batch: [
            (seqC, theta, probR),
            (seqC, theta, probR),
            ...
            (seqC, theta, probR),
            ]
            seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
            
            shuffling_method: 0: complex shuffle - expand x from (B, D*M*S, 16) to (B*C, D*M*S, 16) then shuffle along the 2nd axis, then shuffle the batch BC
                              1: simple shuffle - shuffle x (B, D*M*S, 16) along the 2nd axis, then expand x to (B*C, D*M*S, 16)
    """
    
    C = config['dataset']['num_probR_sample']
    B = len(batch)
    
    if debug:
        start_time_0 = time.time()
    
    # Preallocate tensors
    seqC_batch = torch.empty((B, *batch[0][0].shape))
    theta_batch = torch.empty((B, *batch[0][1].shape))
    probR_batch = torch.empty((B, *batch[0][2].shape))
    
    # Fill tensors with data from the batch
    for i, (seqC, theta, probR) in enumerate(batch):
        seqC_batch[i] = seqC
        theta_batch[i] = theta
        probR_batch[i] = probR
        
    # seqC, theta, probR = zip(*batch)
    
    # seqC_batch = torch.stack(seqC)
    # theta_batch = torch.stack(theta)
    # probR_batch = torch.stack(probR)
    
    if debug:
        print(f"collate_fn_vec: dataloading {(time.time() - start_time_0)*1000:.2f} ms")
    
    del batch, seqC, theta, probR
    
    if shuffling_method == 0:
        # Repeat seqC and theta C times along a new dimension
        seqC_batch = seqC_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 15) first C samples are the same, from the first batch
        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)
        
        # Repeat probR C times along a new dimension and sample from Bernoulli distribution
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        # probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)
        # probR_batch.bernoulli_() # (C*B, D*M*S, 1)
        
        # Concatenate x_seqC and x_choice
        x_batch = torch.cat([seqC_batch, probR_batch.bernoulli_()], dim=-1)  # (C*B, D*M*S, 16)
        del probR_batch, seqC_batch
        gc.collect()
        
        if debug:
            print(f"\ncollate_fn_vec: get x_batch {(time.time() - start_time_0)*1000:.2f} ms")
            start_time = time.time()
            
        # Shuffle x along the 2nd axis
        # x_batch = torch.stack([x_batch[i,:,:][torch.randperm(x_batch.shape[1]),:] for i in range(x_batch.shape[0])])
        DMS = x_batch.shape[1]
        # x_batch_shuffled = torch.empty_like(x_batch)
        
        # permutations = generate_permutations(B*C, DMS)
        # permutations = list(map(lambda _: torch.randperm(DMS), range(B*C)))
        
        # permutations = [torch.randperm(DMS) for _ in range(B*C)]
        permutations = torch.stack([torch.randperm(DMS) for _ in range(B*C)])
        # permutations = torch.rand(B*C, DMS).argsort(dim=-1)
        
        # if debug:
        #     print(f"\ncollate_fn_vec: generate permutations {(time.time() - start_time)*1000:.2f} ms")
        #     start_time = time.time()
        # start_time = time.time()
        
        # for i in range(B*C):
            # x_batch_shuffled[i] = x_batch[i][permutations[i]]
            # x_batch_shuffled[i] = x_batch[i][torch.randperm(DMS)]
        
        # gathering method
        # indices = torch.argsort(torch.rand(*x_batch.shape[:2]), dim=1)
        # x_batch_shuffled = torch.gather(x_batch, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, x_batch.shape[-1]))
        
        # del x_batch
        
        if debug:
            print(f"collate_fn_vec: shuffle x_batch {(time.time() - start_time)*1000:.2f} ms")
            start_time = time.time()
            
            # permutations = torch.stack([torch.randperm(DMS) for _ in range(B*C)])
            # x_batch_shuffled = x_batch[torch.arange(B * C)[:, None], permutations]
            # x_batch_shuffled_2 = x_batch[torch.arange(B * C)[:, None], permutations]
            # print(f"collate_fn_vec: shuffle x_batch_2 {(time.time() - start_time)*1000:.2f} ms")
            # # print(f'same? {torch.all(torch.eq(x_batch_shuffled, x_batch_shuffled_2))}')
            # start_time = time.time()
        
        # Shuffle the batched dataset
        # indices             = torch.randperm(x_batch_shuffled.shape[0])
        indices             = torch.randperm(x_batch.shape[0])
        # x_batch_shuffled    = x_batch_shuffled[indices]
        # theta_batch         = theta_batch[indices]
        
        # if debug:
        #     print(f"collate_fn_vec: finish shuffle {(time.time() - start_time)*1000:.2f} ms")
        #     print(f"collate_fn_vec: -- finish computation {(time.time() - start_time_0)*1000:.2f} ms")
        
        # return x_batch_shuffled[indices], theta_batch[indices]
        # shuffle along the 1st axis individually and then shuffle the batch
        # return x_batch[torch.arange(B * C)[:, None], permutations][indices], theta_batch[indices]
        return x_batch[indices[:, None], permutations], theta_batch[indices]
    
    elif shuffling_method == 1:
        
        # shuffle seqC_batch and theta_batch along the 2nd axis
        DMS         = seqC_batch.shape[1]
        for i in range(B):
            indices     = torch.randperm(DMS)
            seqC_batch[i]  = seqC_batch[i][indices]
            probR_batch[i] = probR_batch[i][indices]
        
        theta_batch = theta_batch.repeat_interleave(C, dim=0)  # (C*B, 4)
        seqC_batch  = seqC_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 15)
        probR_batch = probR_batch.repeat_interleave(C, dim=0)  # (C*B, D*M*S, 1)
        probR_batch = torch.bernoulli(probR_batch)  # (C*B, D*M*S, 1)
        
        x_batch = torch.cat([seqC_batch, probR_batch], dim=-1)  # (C*B, D*M*S, 16)
        del seqC_batch, probR_batch
        
        return x_batch, theta_batch
        
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