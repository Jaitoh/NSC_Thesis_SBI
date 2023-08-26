import time
import argparse
import torch
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import sys
import yaml
from sbi import analysis
import os
from scipy.stats import gaussian_kde
from tqdm import tqdm

NSC_DIR = Path(os.getcwd()).resolve().parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")
print(NSC_DIR)
from utils.setup import adapt_path
from utils.event import get_train_valid_lr
from utils.plots import load_img, pairplot, plot_posterior_mapped_samples, marginal_plot
from utils.inference import (
    get_posterior,
    load_stored_config,
    sampling_from_posterior,
    ci_perf_on_dset,
    perfs_on_dset,
)
from utils.train import WarmupScheduler, plot_posterior_with_label, load_net, get_limits
from simulator.model_sim_pR import DM_sim_for_seqCs_parallel_with_smaller_output
from utils.range import x2seqC, seqC2x, convert_samples_range

import matplotlib as mpl
import matplotlib.pyplot as plt

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
# remove all edges

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE, weight="bold")  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE, labelweight="bold")  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams["axes.edgecolor"] = "k"
mpl.rcParams["axes.linewidth"] = 2

font = {"weight": "bold"}
mpl.rc("font", **font)

# grid alpha to 0.2
mpl.rcParams["grid.alpha"] = 0.2

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
from features.features import *

# load the data
fig_dir = Path(f"{NSC_DIR}/codes/notebook/figures/")
valid_data_dir = f"{fig_dir}/compare/dataset_varying_params.pt"
data = torch.load(valid_data_dir)
x_o = data["x_o"]
seqC_o = data["seqC_o"]
params = data["params"]
probR = data["probR"]
chR = data["chR"]
prior_labels = data["prior_labels"]
normed_limits = data["normed_limits"]
designed_limits = data["designed_limits"]
step = 7
nT = 28
C_idx = 0
D, M, S = seqC_o.shape[0], seqC_o.shape[1], seqC_o.shape[2]
DMS = D * M * S

# map 3, 5, 7, 9, 11, 13, 15
chosen_dur_list = np.array([3, 9, 15])
chosen_dur_idx = ((chosen_dur_list - 3) / 2).astype(int)

x_o_chosen_dur = x_o[chosen_dur_idx].reshape(-1, 15)
x_o_all = x_o.reshape(-1, 15)
print(f"==>> x_o_chosen_dur.shape: {x_o_chosen_dur.shape}")
print(f"==>> x_o_all.shape: {x_o_all.shape}")


def main():
    # get PID
    pid = os.getpid()
    print(f"\n==>> pid: {pid}")

    start_time = time.time()
    pipeline_version = "nle-p2"
    train_id = "L0-nle-p2-cnn"
    exp_id = "L0-nle-p2-cnn-datav2"
    # exp_id = "L0-nle-p2-cnn-datav2-small-batch-tmp"
    log_exp_id = "nle-p2-cnn-datav2"
    use_chosen_dur = 0
    T_idx = 0
    iid_batch_size_theta = 500

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_version", type=str, default=pipeline_version)
    parser.add_argument("--train_id", type=str, default=train_id)
    parser.add_argument("--exp_id", type=str, default=exp_id)
    parser.add_argument("--log_exp_id", type=str, default=log_exp_id)
    parser.add_argument("--use_chosen_dur", type=int, default=use_chosen_dur)
    parser.add_argument("--T_idx", type=int, default=T_idx)
    parser.add_argument("--iid_batch_size_theta", type=int, default=iid_batch_size_theta)
    args = parser.parse_args()

    pipeline_version = args.pipeline_version
    train_id = args.train_id
    exp_id = args.exp_id
    log_exp_id = args.log_exp_id
    use_chosen_dur = args.use_chosen_dur
    T_idx = args.T_idx
    iid_batch_size_theta = args.iid_batch_size_theta

    # == load the latest event file
    log_dir = Path(NSC_DIR) / "codes/src/train_nle/logs" / train_id / exp_id

    config_nle, model_path_nle = load_stored_config(exp_dir=log_dir)

    if "p2" in pipeline_version:
        from train_nle.train_p2 import Solver
    if "p3" in pipeline_version:
        from train_nle.train_p3 import Solver

    solver_nle = Solver(config_nle, store_config=False)
    solver_nle.init_inference(
        iid_batch_size_x=config_nle.posterior.MCMC_iid_batch_size_x,  #!
        # iid_batch_size_theta=config_nle.posterior.MCMC_iid_batch_size_theta,  # + info: 10000 MCMC init, other time 1
        iid_batch_size_theta=iid_batch_size_theta,  # + info: 10000 MCMC init, other time 1
        sum_writer=False,
    )

    # get the trained network
    _, _, density_estimator, _, _ = solver_nle.inference.prepare_dataset_network(
        config_nle,
        continue_from_checkpoint=model_path_nle,
        device="cuda" if torch.cuda.is_available() and config_nle.gpu else "cpu",
        print_info=True,
        inference_mode=True,
        low_batch=5,
    )

    # build the posterior
    mcmc_parameters = dict(
        warmup_steps=config_nle.posterior.warmup_steps,  #!
        thin=config_nle.posterior.thin,  #!
        # num_chains=min(os.cpu_count() - 1, config_nle.posterior.num_chains),  #!
        num_chains=4,  #!
        # num_chains=1,
        # num_workers=config_nle.posterior.num_workers,  #!
        # num_workers=4,  #!
        num_workers=1,  #! for test
        init_strategy="sir",  #!
    )
    print(f"==>> mcmc_parameters: {mcmc_parameters}")

    posterior_nle = solver_nle.inference.build_posterior(
        density_estimator=density_estimator,
        prior=solver_nle.inference._prior,
        sample_with="mcmc",  #!
        mcmc_method="slice",  #!
        # mcmc_method="slice_np",
        # mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_parameters,  #!
        # sample_with="vi",
        # vi_method="rKL",
        # vi_parameters={},
        # rejection_sampling_parameters={},
    )

    # == posterior inference
    # for T in range(nT):
    for T in [T_idx]:
        # skip the first and last step cases
        if T % step == 0 or T % step == step - 1:
            continue

        # which theta is moving
        moving_theta_idx = T // step
        trial_idx = T % step - 1

        # == prepare the data for inference
        # = convert the theta to the normed range
        theta_test = torch.tensor(params[T, :]).clone().detach()
        theta_test = convert_samples_range(theta_test, designed_limits, normed_limits)

        if use_chosen_dur:
            print(use_chosen_dur, "use chosen dur")
            xy_o_chosen_dur = torch.cat(
                [x_o[chosen_dur_idx], chR[chosen_dur_idx, :, :, T, C_idx, None]], dim=-1
            ).reshape(-1, 16)
            xy_o_chosen_dur = xy_o_chosen_dur[:, 1:]
            xy_o = xy_o_chosen_dur
        else:
            print(use_chosen_dur, "use all dur")
            xy_o_all = torch.cat((x_o, chR[:, :, :, T, C_idx, None]), dim=-1).reshape(-1, 16)
            xy_o_all = xy_o_all[:, 1:]
            xy_o = xy_o_all

        print(f"==>> xy_o.shape: {xy_o.shape}")
        print("start posterior inference")

        samples = sampling_from_posterior(
            "cuda",
            posterior_nle,
            xy_o,
            num_samples=2000,
            show_progress_bars=True,
        )

        # save the samples
        if use_chosen_dur:
            save_dir = f"{fig_dir}/compare/{log_exp_id}_posterior_samples_T{T}_chosen_dur.npy"
        else:
            save_dir = f"{fig_dir}/compare/{log_exp_id}_posterior_samples_T{T}_all.npy"

        np.save(save_dir, samples)
        print(f"==>> saved samples: {save_dir}")

    print(f"==>> Done in {(time.time() - start_time)/60/60:.2f} hours")


if __name__ == "__main__":
    main()
