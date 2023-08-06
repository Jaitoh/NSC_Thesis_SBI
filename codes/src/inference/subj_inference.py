"""inference with subject data

load posterior
load and process trial data
inference with trial data
"""
import torch
from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.subject import get_xo
from utils.inference import (
    load_stored_config,
    get_posterior,
    estimate_theta_from_post_samples,
    convert_normed_theta,
)
from utils.train import plot_posterior_unseen


def estimate_theta_for_usr_npe(subj_ID, exp_dir):
    # load config
    config, model_path = load_stored_config(exp_dir=exp_dir)
    if "p4" in config.pipeline_version:
        from train.train_L0_p4a import Solver
    if "p5" in config.pipeline_version:
        from train.train_L0_p5a import Solver

    # load trained posterior model
    solver, posterior = get_posterior(
        model_path=model_path,
        config=config,
        device="cuda",
        Solver=Solver,
        low_batch=10,
    )

    # load and process trial data
    data_path = Path(NSC_DIR) / "data/trials.mat"
    seqC, chR = get_xo(
        data_path,
        subj_ID=subj_ID,
        dur_list=config.dataset.chosen_dur_list,
        MS_list=[0.2, 0.4, 0.8],
    )
    xo = torch.cat((seqC, chR), dim=1).float().to("cuda")

    # inference with trial data
    prior_limits = solver._get_limits()
    fig, axes, samples = plot_posterior_unseen(
        posterior=posterior,
        sample_num=2000,
        x=xo,
        limits=prior_limits,
        prior_labels=config.prior.prior_labels,
        show_progress_bars=True,
    )

    # estimate with posterior samples
    theta_estimated_normed = estimate_theta_from_post_samples(
        prior_limits=prior_limits,
        samples=samples,
    )

    theta_estimated, original_range_pair = convert_normed_theta(
        theta_estimated_normed=theta_estimated_normed,
        prior_min=config.prior.prior_min,
        prior_max=config.prior.prior_max,
        prior_labels=config.prior.prior_labels,
    )

    print("\n" + "-" * 20 + f" theta_estimated for subject {subj_ID} " + "-" * 20)
    for label, estimate, range_ in zip(config.prior.prior_labels, theta_estimated, original_range_pair):
        print(f"{label:10}: {estimate:8.3f} from range [{range_[0]:.3f}, {range_[1]:.3f}]")

    return theta_estimated


if __name__ == "__main__":
    subj_ID = 2
    exp_dir = "~/tmp/NSC/codes/src/train/logs/train_L0_p5a/p5a-conv_net-tmp"

    theta_estimated = estimate_theta_for_usr_npe(subj_ID, exp_dir)