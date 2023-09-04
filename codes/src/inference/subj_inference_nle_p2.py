"""inference with subject data

load posterior
load and process trial data
inference with trial data
"""
import torch
import pickle
import argparse
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
from utils.setup import adapt_path


def estimate_theta_for_usr_nle(subj_ID, exp_dir):
    # load config
    config, model_path = load_stored_config(exp_dir=exp_dir)
    subj_post_samples_path = f"{exp_dir}/posterior/samples_obs_Subject{subj_ID}.pt"

    prior_min = config.prior.prior_min
    prior_max = config.prior.prior_max
    prior_limits = [[x, y] for x, y in zip(prior_min, prior_max)]

    samples = torch.load(subj_post_samples_path)

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

    return theta_estimated, original_range_pair, config.prior.prior_labels


if __name__ == "__main__":
    exp_dir = "~/tmp/NSC/codes/src/train_nle/logs/L0-nle-p3-cnn/L0-nle-p3-cnn-newLoss"
    exp_dir = adapt_path(exp_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, default=exp_dir, help="log directory")
    args = parser.parse_args()

    exp_dir = adapt_path(args.exp_dir)

    subj_thetas = {}
    subj_thetas["exp"] = str(exp_dir).split("/")[-2:]
    for subj_ID in range(2, 13):
        theta_estimated, range_, labels = estimate_theta_for_usr_nle(subj_ID, exp_dir)
        subj_thetas["range_"] = range_
        subj_thetas["labels"] = labels
        subj_thetas[subj_ID] = theta_estimated

    Path(f"{exp_dir}/inference").mkdir(parents=True, exist_ok=True)
    # save pkl
    with open(f"{exp_dir}/inference/subj_thetas.pkl", "wb") as f:
        pickle.dump(subj_thetas, f)
    # save txt
    with open(f"{exp_dir}/inference/subj_thetas.txt", "w") as f:
        for subj_ID in range(2, 13):
            f.write(f"subj_ID={subj_ID}\n")
            for label, estimate in zip(labels, subj_thetas[subj_ID]):
                f.write(f"{label:10}: {estimate:8.3f}\n")
            f.write("\n")
