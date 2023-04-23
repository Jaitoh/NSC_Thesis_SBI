import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from dataset.data_process import process_x_seqC_part

class MyDataset(Dataset):
    """MyDataset class

    Returns:
        seqC : (DMS, 15)
        theta: (4,)
        probR: (DMS, 1)
    """
    def __init__(self, 
                 data_path, 
                 num_theta_each_set=5000, 
                 seqC_process_method='norm',
                 nan2num=-1,
                 summary_type=0):
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as f:
            self.total_sets = len(f.keys())  # Total number of groups (sets)
            self.total_samples = num_theta_each_set*self.total_sets
            self.num_theta_each_set = num_theta_each_set
            self.seqC_process_method = seqC_process_method
            self.nan2num = nan2num
            self.summary_type = summary_type
                        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        set_idx, theta_idx = divmod(idx, self.num_theta_each_set)
        
        # Load seqC, theta, and probR for the given idx
        with h5py.File(self.data_path, 'r') as f:
            seqC = f[f'set_{set_idx}']['seqC'][:]
            seqC = seqC.reshape((seqC.shape[0]*seqC.shape[1]*seqC.shape[2], seqC.shape[3]))
            # print(f"seqC shape: {seqC.shape}")
            theta = f[f'set_{set_idx}']['theta'][theta_idx,:]
            # print(f"theta shape: {theta.shape}")
            probR = f[f'set_{set_idx}']['probR'][..., theta_idx, :]
            probR = probR.reshape((probR.shape[0]*probR.shape[1]*probR.shape[2], probR.shape[3]))
            # print(f"probR shape: {probR.shape}")

        seqC = process_x_seqC_part(
            seqC                = seqC, 
            seqC_process_method = self.seqC_process_method,
            nan2num             = self.nan2num,
            summary_type        = self.summary_type,
            )
        
        return seqC, theta, probR
    


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, C, shuffle=True, pin_memory=True, num_workers=0):
        super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=self.collate_fn, pin_memory=pin_memory, num_workers=num_workers)
        self.C = C
        
    def collate_fn(self, batch):
        # Process the batch
        x_batch, theta_batch = [], []
        
        x_batch     = torch.empty((self.C * len(batch), batch[0][0].shape[0], batch[0][0].shape[1]+batch[0][2].shape[1]))
        theta_batch = torch.empty((self.C * len(batch), batch[0][1].shape[0]))
        
        for i, (seqC, theta, probR) in enumerate(batch): # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
            probR    = torch.tensor(probR).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, D*M*S, 1)
            x_choice = torch.bernoulli(probR) # (C, D*M*S, 1)
            
            x_seqC   = torch.tensor(seqC).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, D*M*S, 15)
            x = torch.cat([x_seqC, x_choice], dim=-1)
            
            # Shuffle x along the 2nd axis
            for c in range(self.C):
                indices = torch.randperm(x.shape[1])
                x[c] = x[c][indices]
            
            theta = torch.tensor(theta).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, 4)
            
            x_batch[i*self.C:(i+1)*self.C] = x
            theta_batch[i*self.C:(i+1)*self.C] = theta
            
        # Shuffle the batched dataset
        indices     = torch.randperm(x_batch.shape[0])
        x_batch     = x_batch[indices]
        theta_batch = theta_batch[indices]
        
        return torch.tensor(x_batch, dtype=torch.float32), torch.tensor(theta_batch, dtype=torch.float32)    



def collate_fn_probR(batch, Rchoice_method='probR_sampling', num_probR_sample=10):
    '''
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
