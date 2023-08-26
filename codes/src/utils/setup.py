import os
from pathlib import Path
import argparse
import gc
import torch


def adapt_path(input_path):
    """
    input_path: str or Path
        e.g. ~/tmp/NSC/data/dataset_L0_exp_set_0.h5
        e.g. /home/username/tmp/NSC/data/dataset_L0_exp_set_0.h5
        e.g. home/username/tmp/NSC/data/dataset_L0_exp_set_0.h5
        e.g. ./src/username/tmp/NSC/data/dataset_L0_exp_set_0.h5

    output_path: replace with correct username
    """
    home_dir = Path.home()
    if "home" in str(input_path):
        relative_path = str(input_path).split("home")[1].split("/")[2:]
        # output_path = Path("/".join(["~"] + relative_path)).expanduser()
        output_path = Path("/".join([str(home_dir)] + relative_path))
    elif "~" in str(input_path):
        # output_path = Path(input_path).expanduser()
        relative_path = str(input_path).split("~")[1:]
        # print(f"relative_path: {relative_path}")
        output_path = Path("/".join([str(home_dir)] + relative_path))
    else:
        output_path = Path(input_path)
    
    # replace tmp with data
    if "/tmp/" in str(output_path):
        output_path = Path(str(output_path).replace("/tmp/", "/data/"))

    return output_path


print(adapt_path("./src/username/tmp/NSC/data/dataset_L0_exp_set_0.h5"))


def torch_var_size(var, unit="KB"):
    if unit == "KB":
        size_ = var.element_size() * var.nelement() // 1024
        size_ = f"{size_}KB"

    elif unit == "MB":
        size_ = var.element_size() * var.nelement() // 1024 // 1024
        size_ = f"{size_}MB"
    else:
        raise ValueError("unit must be KB or MB")

    return size_


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


def report_memory():
    total_mem = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                mem = obj.element_size() * obj.nelement()
                total_mem += mem
                print(f"Object: {obj}, occupies {mem} bytes on GPU")
        except Exception as e:
            pass
    print(f"Total memory occupied by tensors: {total_mem} bytes")


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
    parser.add_argument("--data_path", type=str, default=data_path, help="simulated data store/load dir")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./src/train/logs/log_test",
        help="training log dir",
    )
    parser.add_argument("--gpu", action="store_false", help="Use GPU by default.")
    # parser.add_argument('--finetune', type=str, default=None, help='Load model from this job for finetuning.')
    parser.add_argument("--eval", action="store_true", help="Evaluation mode.")
    parser.add_argument("-y", "--overwrite", action="store_false", help="Overwrite log dir by default.")
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
    log_dir = adapt_path(log_dir)
    data_path = adapt_path(data_path)

    print(f"\n--- dir settings ---")
    print(f"log dir: {str(log_dir)}")
    print(f"data dir: {str(data_path)}")

    model_dir = log_dir / "model"
    posterior_dir = log_dir / "posterior"
    posterior_figures_dir = log_dir / "posterior" / "figures"
    # event_fig_dir = log_dir / "event_fig"
    # event_hist_dir = log_dir / "event_hist"

    # make dirs
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    if not adapt_path(model_dir).exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    if not adapt_path(posterior_dir).exists():
        posterior_dir.mkdir(parents=True, exist_ok=True)
    if not adapt_path(posterior_figures_dir).exists():
        posterior_figures_dir.mkdir(parents=True, exist_ok=True)
        # event_fig_dir.mkdir(parents=True, exist_ok=True)
        # event_hist_dir.mkdir(parents=True, exist_ok=True)

    # elif log_dir.exists():
    #     remove_files_except_resource_log(log_dir)
    #     if not model_dir.exists():
    #         os.makedirs(str(model_dir))
    #     if not posterior_dir.exists():
    #         os.makedirs(str(posterior_dir))
    #     if not posterior_figures_dir.exists():
    #         os.makedirs(str(posterior_figures_dir))
    #     if not event_fig_dir.exists():
    #         os.makedirs(str(event_fig_dir))
    #     if not event_hist_dir.exists():
    #         os.makedirs(str(event_hist_dir))

    # remove events.out.tfevents files from log_dir
    for root, dirs, files in os.walk(log_dir):
        for name in files:
            if name.startswith("events.out.tfevents"):
                file_path = os.path.join(root, name)
                os.remove(file_path)

    # check data path, where to read the data from, exists
    if not data_path.exists():
        assert False, f"Data dir {str(data_path)} does not exist."
