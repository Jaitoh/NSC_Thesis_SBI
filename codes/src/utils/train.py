import argparse
import torch

def get_args():
    """
    Returns:
        args: Arguments
    """
    parser = argparse.ArgumentParser(description='pipeline for sbi')
    # parser.add_argument('--run_test', action='store_true', help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    # parser.add_argument('--run_simulator', type=int, default=0, help="""run simulation to generate dataset and store to local file
    #                                                                     0: no simulation, load file directly and do the training
    #                                                                     1: run simulation and do the training afterwards
    #                                                                     2: only run the simulation and do not train""")
    parser.add_argument('--config_simulator_path', type=str, default="./src/config/test/test_simulator.yaml",
                        help="Path to config_simulator file")
    parser.add_argument('--config_dataset_path', type=str, default="./src/config/test/test_dataset.yaml",
                        help="Path to config_train file")
    parser.add_argument('--config_train_path', type=str, default="./src/config/test/test_train.yaml",
                        help="Path to config_train file")
    # parser.add_argument('--data_dir', type=str, default="../data/train_datas/",
    #                     help="simulated data store/load dir")
    parser.add_argument('--log_dir', type=str, default="./src/train/logs/log_test", help="training log dir")
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
    parser.add_argument('-y', '--overwrite', action='store_true', help='Overwrite log dir.')
    args = parser.parse_args()

    return args



def print_cuda_info(device):
    """
    Args:
        device: 'cuda' or 'cpu'
    """
    if device == 'cuda':
        print('--- CUDA info ---')
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

        torch.cuda.memory_summary(device=None, abbreviated=False)