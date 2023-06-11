"""
Compare the dataset generated using function 
 - simulate_for_sbi
 - simulate_and_save
"""
import hydra
import torch
import sys
sys.path.append('./src')
from dataset.dataset import training_dataset
from dataset.simulate_for_sbi import simulate_for_sbi
from dataset.simulate_and_save import simulate_and_save
from simulator.model_sim_pR import get_boxUni_prior
from utils.set_seed import setup_seed

@hydra.main(config_path="../../src/config", config_name="config-test-dataset", version_base=None)
def main(config):
    
    prior = get_boxUni_prior(
        prior_min=config['prior']['prior_min'],
        prior_max=config['prior']['prior_max'],
        )
    proposal = prior
    
    data_path = './data_tmp.h5'
    seqC_a, theta_a, probR_a = simulate_for_sbi(proposal, config, seed=0, debug=True)
    seqC_b, theta_b, probR_b = simulate_and_save(data_path, config, seed=0)

    if (probR_a == probR_b).all():
        print('generated seqC, theta, probR are the same for both functions')
    
    # check the processing method
    dataset = training_dataset(config)
    x, theta = dataset.data_process_pipeline(
        seqC_a, theta_a, probR_a,
    )
    theta_a = theta.clone().detach().to(torch.float32) # avoid float64 error
    x_a     = x.clone().detach().to(torch.float32)
    
    # method 2 - get one sample on the fly
    the_shape = seqC_b.shape
    seqC_b = seqC_b.reshape(the_shape[0]*the_shape[1]*the_shape[2], the_shape[3]) # DMS, 15
    choice_b = probR_b.repeat_interleave(25, dim=-1)
    choice_b = torch.bernoulli(choice_b)
    x_b = torch.empty((the_shape[0]*the_shape[1]*the_shape[2], 16))
    x_b[:, 0:15] = seqC_b
    x_b[:, 15] = choice_b
    theta_b = theta_b[0,:]
    torch.sum(x_a[0,:,:]-x_b[:,:]) # should be zero
    
    
    
    # compare permute and reshape
    # import torch
    # D, M, S, T, C = 3, 3, 700, 10, 25
    # a = torch.rand((D, M, S, 15))
    # a_0 = a.permute(3, 0, 1, 2).view(15, D*M*S)
    # a_0 = a_0.permute(1, 0)
    # a_1 = a.reshape(D*M*S, 15)
    # print(torch.sum(a_0-a_1)) # should be zero
    
if __name__ == '__main__':
    main()