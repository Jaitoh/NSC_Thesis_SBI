
import sys
sys.path.append('./src')

from config.load_config import load_config
from pathlib import Path
import time
from dataset.seqC_generator import seqC_generator
from dataset.model_sim_pR import get_boxUni_prior, DM_sim_for_seqCs_parallel
from dataset.dataset_pipeline import training_dataset


def simulate_for_sbi(proposal, config, run, save_sim_data, save_train_data):
    
    tic = time.time()
    seqC = seqC_generator(nan_padding=None).generate(
        dur_list = config['x_o']['chosen_dur_list'],
        MS_list=config['x_o']['chosen_MS_list'],
        seqC_sample_per_MS=config['x_o']['seqC_sample_per_MS'],
    )
    print(f'---\nseqC generated in {(time.time()-tic)/60:.2f}min')

    if save_sim_data:
        save_data_path = Path(config['data_dir']) / f"{config['simulator']['save_name']}_run{run}.h5"
    else:
        save_data_path = None
    
    tic = time.time()
    seqC, theta, probR = DM_sim_for_seqCs_parallel(
            seqCs = seqC,
            prior=proposal,
            num_prior_sample=config['prior']['num_prior_sample'],
            model_name=config['simulator']['model_name'],
            save_data_path=save_data_path,
    )
    print(f'---\nDM parallel simulation finished in {(time.time()-tic)/60:.2f}min')
    
    if save_train_data:
        save_data_path = Path(config['data_dir']) / f"{config['dataset']['save_name']}_run{run}.h5"
    else:
        save_data_path = None
    
    tic = time.time()
    dataset = training_dataset(config)
    x, theta = dataset.data_process_pipeline(
        seqC, theta, probR,
        save_data_path=save_data_path
    )
    print(f'---\nx, theta processing finished in {(time.time()-tic)/60:.2f}min')
    
    return x, theta


if __name__ == '__main__':

    config = load_config(
        config_simulator_path=Path('./src/config') / 'test' / 'test_simulator.yaml',
        config_dataset_path=Path('./src/config') / 'test' / 'test_dataset.yaml',
        config_train_path=Path('./src/config') / 'test' / 'test_train.yaml',
    )
    
    # initialize prior and proposal
    prior = get_boxUni_prior(
        prior_min=config['prior']['prior_min'],
        prior_max=config['prior']['prior_max'],
    )
    proposal = prior
    
    run = 0
    theta, x = simulate_for_sbi(
        proposal,
        config,
        run,
        save_sim_data=True,
        save_train_data=True,
    )