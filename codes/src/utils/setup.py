import os
from pathlib import Path
import argparse

def get_args():
    """
    Returns:
        args: Arguments
    """
    parser = argparse.ArgumentParser(description='pipeline for sbi')
    # parser.add_argument('--run_test', action='store_true', help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--config_simulator_path', type=str, default="./src/config/test/test_simulator.yaml",
                        help="Path to config_simulator file")
    parser.add_argument('--config_dataset_path', type=str, default="./src/config/test/test_dataset.yaml",
                        help="Path to config_train file")
    parser.add_argument('--config_train_path', type=str, default="./src/config/test/test_train.yaml",
                        help="Path to config_train file")
    parser.add_argument('--data_path', type=str, default="../data/dataset/dataset_L0_exp_set_0_test.h5",
                        help="simulated data store/load dir")
    parser.add_argument('--log_dir', type=str, default="./src/train/logs/log_test", help="training log dir")
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
    parser.add_argument('-y', '--overwrite', action='store_true', help='Overwrite log dir.')
    parser.add_argument('--continue_from_checkpoint', type=str, default="", help='continue the training from checkpoint')
    
    parser.add_argument('--run', type=int, default=0, help='run of Round0')
    args = parser.parse_args()

    return args


def remove_files_except_resource_log(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            # if not (name.startswith('training_dataset') or name.startswith('x') or name.startswith('theta') or name.startswith('resource_usage')):
            if not name.startswith('resource_usage'):
                file_path = os.path.join(root, name)
                os.remove(file_path)
                
                
def check_path(log_dir, data_path, args):
    """
    check the path of log_dir and data_dir
    """

    print(f'\n--- dir settings ---\nlog dir: {str(log_dir)}')
    print(f'data dir: {str(data_path)}')

    model_dir = log_dir / 'model'
    training_dataset_dir = log_dir / 'training_dataset'
    posterior_dir = log_dir / 'posterior'
    posterior_figures_dir = log_dir / 'posterior' / 'figures'
    # check log path: if not exists, create; if exists, remove or a fatal error
    if not log_dir.exists():
        os.makedirs(str(log_dir))
        os.makedirs(f'{str(log_dir)}/model/')
        os.makedirs(f'{str(log_dir)}/training_dataset/')
        os.makedirs(f'{str(log_dir)}/posterior/')
        os.makedirs(f'{str(log_dir)}/posterior/figures/')

    elif log_dir.exists() and not args.eval:
        if args.overwrite:
            remove_files_except_resource_log(log_dir)
            if not model_dir.exists():
                os.makedirs(str(model_dir))
            if not training_dataset_dir.exists():
                os.makedirs(str(training_dataset_dir))
            if not posterior_dir.exists():
                os.makedirs(str(posterior_dir))
            if not posterior_figures_dir.exists():
                os.makedirs(str(posterior_figures_dir))
            
        else:
            assert False, f'Run dir {str(log_dir)} already exists.'

    # check data path, where to read the data from, exists
    if not Path(data_path).exists():
        assert False, f'Data dir {str(data_path)} does not exist.'