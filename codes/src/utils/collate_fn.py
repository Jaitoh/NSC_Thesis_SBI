import torch

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
    # theta, x    = zip(*batch)
    
    theta   = torch.stack(theta)
    x       = torch.stack(x)
    
    if Rchoice_method == 'probR':
        # repeat theta and x for each probR sample
        theta_new   = theta.repeat_interleave(num_probR_sample, dim=0)
        x_new       = x.repeat_interleave(num_probR_sample, dim=0)
        x_seqC      = x_new[:, :, :-1]
        x_probRs    = x_new[:, :, -1].unsqueeze_(dim=2)
        
        # sample Rchoice from probR with Bernoulli
        x_Rchoice   = torch.bernoulli(x_probRs)
        x           = torch.cat((x_seqC, x_Rchoice), dim=2)
        theta       = theta_new
    
    return theta, x, _