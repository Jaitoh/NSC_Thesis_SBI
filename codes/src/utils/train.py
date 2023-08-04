import argparse
import torch
from sbi import analysis
import os
import shutil
from pathlib import Path
import sys

import os
import torch.nn as nn

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.setup import adapt_path


def kaiming_weight_initialization(named_parameters):
    for name, param in named_parameters:
        if "weight_ih" in name:
            # nn.init.xavier_uniform_(param.data)  # Xavier initialization
            nn.init.kaiming_uniform_(param.data)  # Kaiming initialization
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            param.data.fill_(0)


def train_inference_helper(inference, **kwargs):
    return inference.train(**kwargs)


def load_net(continue_from_checkpoint, neural_net, device):
    print(f"loading neural net from '{continue_from_checkpoint}'")
    continue_from_checkpoint = adapt_path(continue_from_checkpoint)

    if str(continue_from_checkpoint).endswith("check_point.pt"):
        neural_net.load_state_dict(torch.load(continue_from_checkpoint, map_location=device).state_dict())

    else:
        neural_net.load_state_dict(torch.load(continue_from_checkpoint, map_location=device))
    return neural_net


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, init_lr, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                self.init_lr + (self.target_lr - self.init_lr) * (self.last_epoch / self.warmup_epochs)
                for _ in self.base_lrs
            ]
        else:
            return self.base_lrs


def plot_posterior_with_label(
    posterior, sample_num, x, true_params, limits, prior_labels, show_progress_bars=True
):
    """plot the posterior distribution of the seen data"""

    samples = posterior.sample((sample_num,), x=x, show_progress_bars=show_progress_bars)

    fig, axes = analysis.pairplot(
        samples.cpu().numpy(),
        limits=limits,
        # ticks=[[], []],
        figsize=(10, 10),
        points=true_params.cpu().numpy(),
        points_offdiag={"markersize": 5, "markeredgewidth": 1},
        points_colors="r",
        labels=prior_labels,
        upper=["kde"],
        diag=["kde"],
    )

    return fig, axes, samples


def plot_posterior_unseen(posterior, sample_num, x, limits, prior_labels, show_progress_bars=True):
    """plot the posterior distribution of the seen data"""

    samples = posterior.sample((sample_num,), x=x, show_progress_bars=show_progress_bars)

    fig, axes = analysis.pairplot(
        samples.cpu().numpy(),
        limits=limits,
        # ticks=[[], []],
        figsize=(10, 10),
        labels=prior_labels,
        upper=["kde"],
        diag=["kde"],
    )

    return fig, axes, samples


def choose_cat_validation_set(x, theta, val_set_size, post_val_set):
    """choose and catenate the validation set from the input x and theta

    Args:
        x           (torch.tensor): shape (TC, DMS, L_x)
        theta       (torch.tensor): shape (TC, L_theta)
        val_set_size  (int): the size of the validation set
        post_val_set (dict): the post validation set has keys: 'x', 'x_shuffled', 'theta',

    """

    # randomly choose val_set_size samples from TC samples
    idx = torch.randperm(x.shape[0])
    idx_val = idx[:val_set_size]

    x_val = x[idx_val, :, :]
    theta_val = theta[idx_val, :]

    # randomize the order of each sequence of each line
    x_val_shuffled = torch.empty_like(x_val)
    for k in range(val_set_size):
        x_temp = x_val[k, :, :]
        idx = torch.randperm(x_temp.shape[0])
        x_val_shuffled[k, :, :] = x_temp[idx, :]  # D*M*S,L_x

    # append to the post validation set
    post_val_set["x"] = torch.cat((post_val_set["x"], x_val), dim=0)
    post_val_set["x_shuffled"] = torch.cat((post_val_set["x_shuffled"], x_val_shuffled), dim=0)
    post_val_set["theta"] = torch.cat((post_val_set["theta"], theta_val), dim=0)

    return post_val_set


def print_cuda_info(device):
    """
    Args:
        device: 'cuda' or 'cpu'
    """
    if device == "cuda":
        print("\n--- CUDA info ---")
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
        print("--- CUDA info ---\n")
        torch.cuda.memory_summary(device=None, abbreviated=False)
