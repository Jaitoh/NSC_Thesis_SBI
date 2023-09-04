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

from utils.subject import get_xo

import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    # print PID
    print(f"==>> PID: {os.getpid()}")

    start_time = time.time()
    pipeline_version = "nle-p3"
    train_id = "L0-nle-p3-cnn"
    exp_id = "L0-nle-p3-cnn-newLoss"
    log_exp_id = "L0-nle-p3-cnn-newLoss"

    iid_batch_size_theta = 500
    num_samples = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj_ID", type=int, default=2)
    parser.add_argument("--pipeline_version", type=str, default=pipeline_version)
    parser.add_argument("--train_id", type=str, default=train_id)
    parser.add_argument("--exp_id", type=str, default=exp_id)
    parser.add_argument("--log_exp_id", type=str, default=log_exp_id)
    parser.add_argument("--iid_batch_size_theta", type=int, default=iid_batch_size_theta)
    parser.add_argument("--num_samples", type=int, default=num_samples)
    args = parser.parse_args()

    pipeline_version = args.pipeline_version
    subj_ID = args.subj_ID
    train_id = args.train_id
    exp_id = args.exp_id
    log_exp_id = args.log_exp_id
    iid_batch_size_theta = args.iid_batch_size_theta
    num_samples = args.num_samples

    log_dir = Path(NSC_DIR) / "codes/src/train_nle/logs" / train_id / exp_id

    config_nle, model_path_nle = load_stored_config(exp_dir=log_dir)

    if "p3" in pipeline_version:
        from train_nle.train_p3 import Solver

    solver_nle = Solver(config_nle, store_config=False)
    solver_nle.init_inference(
        iid_batch_size_x=config_nle.posterior.MCMC_iid_batch_size_x,  #!
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
        num_workers=4,  #!
        # num_workers=1,  #! for test
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

    # prepare observed subject data xy_o
    data_path = Path(NSC_DIR) / "data/trials.mat"
    x_o, chR = get_xo(
        data_path,
        subj_ID=subj_ID,
        dur_list=[3, 5, 7, 9, 11, 13, 15],
        MS_list=[0.2, 0.4, 0.8],
    )
    # x_o [14700, 15], [0~1]
    # chR [14700, 1]

    xy_o = torch.cat([x_o[:, 1:], chR], dim=-1)
    xy_o = xy_o[torch.randperm(xy_o.shape[0])]
    print(f"==>> xy_o.shape: {xy_o.shape}")
    print("start posterior inference")

    samples = sampling_from_posterior(
        "cuda",
        posterior_nle,
        xy_o,
        num_samples=num_samples,
        show_progress_bars=True,
    )

    sample_name = f"posterior/samples_obs_Subject{subj_ID}.pt"
    torch.save(samples, adapt_path(log_dir / sample_name))


if __name__ == "__main__":
    main()
