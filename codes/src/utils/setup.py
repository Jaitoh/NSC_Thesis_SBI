import os
from pathlib import Path
import argparse
import gc
import torch


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


# def get_args_run_from_code():
#     """
#     Returns:
#         args: Arguments
#     """
#     train_file_name = 'train_L0'
#     run_id = 'exp-c0-sub0'

#     config_simulator_path = './src/config/simulator/exp_set_0.yaml'
#     config_dataset_path = './src/config/dataset/dataset-setting-1-sub0.yaml'
#     config_train_path = './src/config/train/train_setting_0.yaml'
#     data_path = '../data/dataset/dataset_L0_exp_set_0.h5'
#     log_dir = "./src/train/logs/train_L0/exp-c0-sub0"

#     parser = argparse.ArgumentParser(description='pipeline for sbi')
#     # parser.add_argument('--run_test', action='store_true', help="")
#     parser.add_argument('--seed', type=int, default=0, help="")
#     parser.add_argument('--config_simulator_path', type=str, default=config_simulator_path,
#                         help="Path to config_simulator file")
#     parser.add_argument('--config_dataset_path', type=str, default=config_dataset_path,
#                         help="Path to config_train file")
#     parser.add_argument('--config_train_path', type=str, default=config_train_path,
#                         help="Path to config_train file")
#     parser.add_argument('--data_path', type=str, default=data_path,
#                         help="simulated data store/load dir")
#     parser.add_argument('--log_dir', type=str, default=log_dir, help="training log dir")
#     parser.add_argument('--gpu', action='store_false', help='Use GPU.')
#     # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
#     parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
#     parser.add_argument('-y', '--overwrite', action='store_false', help='Overwrite log dir.')
#     parser.add_argument('--continue_from_checkpoint', type=str, default="", help='continue the training from checkpoint')

#     parser.add_argument('--run', type=int, default=0, help='run of Round0')
#     args = parser.parse_args()

#     return args


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_args(
    config_simulator_path="./src/config/test/test_simulator.yaml",
    config_dataset_path="./src/config/test/test_dataset.yaml",
    config_train_path="./src/config/test/test_train.yaml",
    data_path="../data/dataset/dataset_L0_exp_set_0_test.h5",
):
    """
    Returns:
        args: Arguments
    """
    parser = argparse.ArgumentParser(description="pipeline for sbi")
    # parser.add_argument('--run_test', action='store_true', help="")
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument(
        "--config_simulator_path",
        type=str,
        default=config_simulator_path,
        help="Path to config_simulator file",
    )
    parser.add_argument(
        "--config_dataset_path",
        type=str,
        default=config_dataset_path,
        help="Path to config_train file",
    )
    parser.add_argument(
        "--config_train_path",
        type=str,
        default=config_train_path,
        help="Path to config_train file",
    )
    parser.add_argument(
        "--data_path", type=str, default=data_path, help="simulated data store/load dir"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./src/train/logs/log_test",
        help="training log dir",
    )
    parser.add_argument("--gpu", action="store_false", help="Use GPU by default.")
    # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
    parser.add_argument("--eval", action="store_true", help="Evaluation mode.")
    parser.add_argument(
        "-y", "--overwrite", action="store_false", help="Overwrite log dir by default."
    )
    parser.add_argument(
        "--continue_from_checkpoint",
        type=str,
        default="",
        help="continue the training from checkpoint",
    )

    parser.add_argument("--run", type=int, default=0, help="run of Round0")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    return args


def remove_files_except_resource_log(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            # if not (name.startswith('training_dataset') or name.startswith('x') or name.startswith('theta') or name.startswith('resource_usage')):
            # if not name.startswith('resource_usage') and not name.endswith('log'):
            if not name.endswith("log") and not name.endswith("yaml"):
                file_path = os.path.join(root, name)
                os.remove(file_path)


def check_path(log_dir, data_path):
    """
    check the path of log_dir and data_dir
    """

    print(f"\n--- dir settings ---\nlog dir: {str(log_dir)}")
    print(f"data dir: {str(data_path)}")

    model_dir = log_dir / "model"
    posterior_dir = log_dir / "posterior"
    posterior_figures_dir = log_dir / "posterior" / "figures"
    event_fig_dir = log_dir / "event_fig"
    event_hist_dir = log_dir / "event_hist"

    if not log_dir.exists():
        os.makedirs(str(log_dir))
        os.makedirs(str(model_dir))
        os.makedirs(str(posterior_dir))
        os.makedirs(str(posterior_figures_dir))
        os.makedirs(str(event_fig_dir))
        os.makedirs(str(event_hist_dir))

    elif log_dir.exists():
        # remove_files_except_resource_log(log_dir)
        if not model_dir.exists():
            os.makedirs(str(model_dir))
        if not posterior_dir.exists():
            os.makedirs(str(posterior_dir))
        if not posterior_figures_dir.exists():
            os.makedirs(str(posterior_figures_dir))
        if not event_fig_dir.exists():
            os.makedirs(str(event_fig_dir))
        if not event_hist_dir.exists():
            os.makedirs(str(event_hist_dir))

    # check data path, where to read the data from, exists
    if not Path(data_path).exists():
        assert False, f"Data dir {str(data_path)} does not exist."
