import time
import h5py
import multiprocessing
import os

import sys
sys.path.append('./src')

from simulator.seqC_generator import seqC_generator
from simulator.model_sim_pR import (
    DM_sim_for_seqCs_parallel_with_smaller_output,
    get_boxUni_prior,    
)
from utils.train import get_args
from utils.resource import monitor_resources
from config.load_config import load_config
from utils.set_seed import setup_seed

def store_info(data_path,dur_list,MS_list,seqC_sample_per_MS,model_name,num_prior_sample,prior_min,prior_max,prior_labels):
    
    with h5py.File(data_path, 'w') as f:
        f.create_dataset('test', data='test')
    print(f'folder/file {data_path} exists, it can be used to store the dataset')

    with h5py.File(data_path, 'w') as f:
        info_group = f.create_group('/info')
        info_group.create_dataset("dur_list", data=dur_list)
        info_group.create_dataset("MS_list", data=MS_list)
        info_group.create_dataset("seqC_sample_per_MS", data=seqC_sample_per_MS)
        info_group.create_dataset("model_name", data=model_name)
        info_group.create_dataset("num_prior_sample", data=num_prior_sample)
        info_group.create_dataset("prior_min", data=prior_min)
        info_group.create_dataset("prior_max", data=prior_max)
        info_group.create_dataset("prior_labels", data=prior_labels)
    
    
def store_data(data_path, seqC, theta, probR):
    with h5py.File(data_path, 'w') as f:
        data_group = f.create_group('/data')
        data_group.create_dataset("seqC", data=seqC)
        data_group.create_dataset("theta", data=theta)
        data_group.create_dataset("probR", data=probR)
    

def simulate_and_save(data_path, config, seed):
    
    dur_list            = config['experiment_settings']['chosen_dur_list']
    MS_list             = config['experiment_settings']['chosen_MS_list']
    seqC_sample_per_MS  = config['experiment_settings']['seqC_sample_per_MS']
    num_prior_sample    = config['prior']['num_prior_sample']
    prior_min           = config['prior']['prior_min']
    prior_max           = config['prior']['prior_max']
    prior_labels        = config['prior']['prior_labels']
    model_name          = config['simulator']['model_name']
    
    setup_seed(seed)
        
    # generate seqC
    tic = time.time()
    seqC = seqC_generator(nan_padding=None).generate(
        dur_list            = dur_list,
        MS_list             = MS_list,
        seqC_sample_per_MS  = seqC_sample_per_MS,
    )
    print(f'seqC generated in {(time.time()-tic)/60:.2f}min')
    
    prior = get_boxUni_prior(
        prior_min=config['prior']['prior_min'],
        prior_max=config['prior']['prior_max'],
    )
    
    # simulate probR with num_prior_sample theta values
    tic = time.time()
    theta, probR = DM_sim_for_seqCs_parallel_with_smaller_output(
            seqCs           = seqC,
            prior           = prior,
            num_prior_sample= num_prior_sample,
            model_name      = model_name,
    )
    print(f'DM parallel simulation finished in {(time.time()-tic)/60:.2f}min')
    
    store_info(data_path, dur_list, MS_list, seqC_sample_per_MS, model_name, num_prior_sample, prior_min, prior_max, prior_labels)
    store_data(data_path, seqC, theta, probR)
    print(f'Results written to the file {data_path}')


def main():
    args = get_args()
    
    PID = os.getpid()
    print(f"PID: {PID}")
    log_file = f"{args.log_dir}/resource_usage_{args.run}.log"
    monitor_process = multiprocessing.Process(target=monitor_resources, args=(PID, 5, log_file))
    monitor_process.start()
    
    try:
        
        config = load_config(
            config_simulator_path=args.config_simulator_path,
            config_dataset_path=args.config_dataset_path,
            config_train_path=args.config_train_path,
        )

        print(f'\n--- args ---')
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

        print('\n--- config keys ---')
        print(config.keys())

        simulate_and_save(args.data_path, config, args.seed)
        
    finally:
        
        monitor_process.terminate()



if __name__ == '__main__':
    main()

