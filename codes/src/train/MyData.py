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
                 seqC_process='norm',
                 nan2num=-1,
                 summary_type=0):
        self.data_path = data_path
        with h5py.File(self.data_path, 'r') as f:
            self.total_sets = len(f.keys())  # Total number of groups (sets)
            self.total_samples = num_theta_each_set*self.total_sets
            self.num_theta_each_set = num_theta_each_set
            self.seqC_process = seqC_process
            self.nan2num = nan2num
            self.summary_type = summary_type
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate set index and theta index within the set
        set_idx, theta_idx = divmod(idx, self.num_theta_each_set)
        
        # Load seqC, theta, and probR for the given idx
        with h5py.File(self.data_path, 'r') as f:
            
            seqC_ = f[f'set_{set_idx}']['seqC'][:]
            seqC = seqC_.reshape((seqC_.shape[0]*seqC_.shape[1]*seqC_.shape[2], seqC_.shape[3]))
            del seqC_
            # print(f"seqC shape: {seqC.shape}")

            theta = f[f'set_{set_idx}']['theta'][theta_idx,:]
            
            # print(f"theta shape: {theta.shape}")
            probR_ = f[f'set_{set_idx}']['probR'][..., theta_idx, :]
            probR = probR_.reshape((probR_.shape[0]*probR_.shape[1]*probR_.shape[2], probR_.shape[3]))
            del probR_
            # print(f"probR shape: {probR.shape}")

        seqC = process_x_seqC_part(
            seqC                = seqC, 
            seqC_process        = self.seqC_process,
            nan2num             = self.nan2num,
            summary_type        = self.summary_type,
            )
        
        # return seqC, theta, probR
        return torch.tensor(seqC, device=self.device), torch.tensor(theta, device=self.device), torch.tensor(probR, device=self.device)
    
    
def collate_fn(batch, C):
    # check cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f" device: {device}")
    # Process the batch
    x_batch, theta_batch = [], []
    
    x_batch     = torch.empty((C * len(batch), batch[0][0].shape[0], batch[0][0].shape[1]+batch[0][2].shape[1]), device=device)
    theta_batch = torch.empty((C * len(batch), batch[0][1].shape[0]), device=device)
    
    for i, (seqC, theta, probR) in enumerate(batch): # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
        # print(seqC.device, theta.device, probR.device)
        probR     = probR.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, D*M*S, 1)
        x_seqC    = seqC.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, D*M*S, 15)
        x_choice  = torch.bernoulli(probR) # (C, D*M*S, 1)
        
        x         = torch.cat([x_seqC, x_choice], dim=-1)
        theta = theta.unsqueeze_(dim=0).repeat_interleave(C, dim=0) # (C, 4)
        
        x_batch[i*C:(i+1)*C] = x
        theta_batch[i*C:(i+1)*C] = theta
        
    # Shuffle x along the 2nd axis
    x_batch = torch.stack([x_batch[i][torch.randperm(x_batch.shape[1])] for i in range(x_batch.shape[0])])
    
    # Shuffle the batched dataset
    indices     = torch.randperm(x_batch.shape[0])
    x_batch     = x_batch[indices]
    theta_batch = theta_batch[indices]
    
    return torch.tensor(x_batch, dtype=torch.float32, device=device), torch.tensor(theta_batch, dtype=torch.float32, device=device)


# class MyDataLoader(DataLoader):
#     def __init__(self, dataset, kwargs):
#         super().__init__(dataset, **kwargs)
#         self.C = kwargs['num_probR_sample']
        
#     def collate_fn(self, batch):
#         # Process the batch
#         x_batch, theta_batch = [], []
        
#         x_batch     = torch.empty((self.C * len(batch), batch[0][0].shape[0], batch[0][0].shape[1]+batch[0][2].shape[1]))
#         theta_batch = torch.empty((self.C * len(batch), batch[0][1].shape[0]))
        
#         for i, (seqC, theta, probR) in enumerate(batch): # seqC: (D*M*S, 15), theta: (4,), probR: (D*M*S, 1)
            
#             probR     = torch.tensor(probR).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, D*M*S, 1)
#             x_seqC    = torch.tensor(seqC).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, D*M*S, 15)
#             x_choice  = torch.bernoulli(probR) # (C, D*M*S, 1)
            
#             x         = torch.cat([x_seqC, x_choice], dim=-1)
            
#             theta = torch.tensor(theta).unsqueeze_(dim=0).repeat_interleave(self.C, dim=0) # (C, 4)
            
#             x_batch[i*self.C:(i+1)*self.C] = x
#             theta_batch[i*self.C:(i+1)*self.C] = theta
        
#         # Shuffle x along the 2nd axis
#         x_batch = torch.stack([x_batch[i][torch.randperm(x_batch.shape[1])] for i in range(x_batch.shape[0])])
        
#         # Shuffle the batched dataset
#         indices     = torch.randperm(x_batch.shape[0])
#         x_batch     = x_batch[indices]
#         theta_batch = theta_batch[indices]
        
#         return torch.tensor(x_batch, dtype=torch.float32), torch.tensor(theta_batch, dtype=torch.float32)    



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
