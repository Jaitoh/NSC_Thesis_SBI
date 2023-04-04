import sys
sys.path.append('./src')
from dataset.dataset_pipeline import process_x_seqC_part
from parse_data.parse_trial_data import parse_trial_data
import warnings
import numpy as np
from config.load_config import load_config
from pathlib import Path

def get_xo(
	subject_id          = 2,
	chosen_dur_list     = [9],
	chosen_MS_list      = [0.2,0.4,0.8],
	seqC_sample_per_MS  = 700,
	trial_data_path     = '../data/trials.mat',
    seqC_process_method = 'norm',
    nan2num             = -1,
    summary_type        = 0,
):
    """extract the x_o from the trials.mat file

    Args:
        subject_id (int, optional): subject id. Defaults to 2.
        chosen_dur_list (list, optional): chosen dur list. Defaults to [9].
        chosen_MS_list (list, optional): chosen MS lsit. Defaults to [0.2,0.4,0.8].
        seqC_sample_per_MS (int, optional): number of seqC samples per MS. Defaults to 700.
        trial_data_path (str, optional): data path store the trials.mat. Defaults to '../../data/trials.mat'.
        --- args for process_x_seqC_part ---
        seqC_process_method (str, optional): seqC process method. Defaults to 'norm'.
        nan2num (int, optional): nan2num. Defaults to -1.
        summary_type (int, optional): summary type. Defaults to 0.
        
    Returns:
        
    """
    trials = parse_trial_data(trial_data_path)
    
    subject_idx = trials['subjID']==subject_id
    subject_idx = subject_idx.reshape(-1)

    seqC_subj = trials['pulse'][subject_idx]
    MS_subj = trials['MS'][subject_idx][:,0]
    ch_subj = trials['chooseR'][subject_idx][:,0]
    dur_subj = trials['dur'][subject_idx][:,0]

    idx_chosen = [i for i in range(len(dur_subj)) if MS_subj[i] in chosen_MS_list and dur_subj[i] in chosen_dur_list]
    seqC_chosen = seqC_subj[idx_chosen]
    ch_chosen = ch_subj[idx_chosen].reshape(-1,1)

    x_seqC_chosen = process_x_seqC_part(
        seqC_chosen, 
        seqC_process_method,
        nan2num,
        summary_type,
    )
    # x_seqC_chosen = seqC_pattern_summary(seqC_chosen, summary_type=0)
    x_o_chosen = np.concatenate([x_seqC_chosen, ch_chosen], axis=-1) # DM*S, L_x+1

    if x_seqC_chosen.shape[0] != seqC_sample_per_MS*len(chosen_dur_list)*len(chosen_MS_list):
        warnings.warn('x_seqC_chosen.shape[0] != seqC_sample_per_MS*len(chosen_dur_list)*len(chosen_MS_list)')
    
    x_o = x_o_chosen
    print('---\nx_o information')
    print('x_o.shape: ', x_o.shape)
    print('subject_id: ', subject_id)
    print('chosen_dur_list: ', chosen_dur_list)
    print('chosen_MS_list: ', chosen_MS_list)
    print('seqC_sample_per_MS: ', seqC_sample_per_MS)
    
    return x_o

if __name__ == '__main__':
    
    test = True

    if test:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'test' / 'test_simulator.yaml',
            config_dataset_path=Path('./src/config') / 'test' / 'test_dataset.yaml',
            config_train_path=Path('./src/config') / 'test' / 'test_train.yaml',
        )
    else:
        config = load_config(
            config_simulator_path=Path('./src/config') / 'simulator' / 'simulator_Ca_Pa_Ma.yaml',
            config_dataset_path=Path('./src/config') / 'dataset' / 'dataset_Sa0_Ra_suba0.yaml',
            config_train_path=Path('./src/config') / 'train' / 'train_Ta0.yaml',
        )
    print(config.keys())
    
    x_o = get_xo(
        subject_id          = config['x_o']['subject_id'],
        chosen_dur_list     = config['x_o']['chosen_dur_list'],
        chosen_MS_list      = config['x_o']['chosen_MS_list'],
        seqC_sample_per_MS     = config['x_o']['seqC_sample_per_MS'],
        trial_data_path     = config['x_o']['trial_data_path'],
        
        seqC_process_method = config['dataset']['seqC_process'],
        nan2num             = config['dataset']['nan2num'],
        summary_type        = config['dataset']['summary_type'],
    )
    