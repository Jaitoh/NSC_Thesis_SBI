from pathlib import Path
from omegaconf import OmegaConf
import sys
import torch
import numpy as np
from sbi import analysis
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.append("./src")
sys.path.append("../../src")

from train.train_L0_p4 import Solver

# compute the distance between the predicted and the ground truth
LOG_DIR = "/home/ubuntu/tmp/NSC/codes/src/train/logs"
EXP_ID = "train_L0_p4/p4-4Fs-1D-cnn"


# === compute one example distance ===
# load config
config_path = Path(LOG_DIR) / EXP_ID / "config.yaml"
model_path = Path(LOG_DIR) / EXP_ID / "model" / "best_model.pt"
config = OmegaConf.load(config_path)
config.log_dir = str(Path(LOG_DIR) / EXP_ID)

# get the trained posterior
solver = Solver(config)
solver.init_inference().prepare_dataset_network(config, model_path, device="cpu")

posterior = solver.inference.build_posterior(solver.inference._neural_net)
solver.inference._model_bank = []

# get input data x (and theta)
x = solver.inference.seen_data_for_posterior["x"][0]
theta = solver.inference.seen_data_for_posterior["theta"][0]

# sample from the posterior with x
samples = posterior.sample((20000,), x=x, show_progress_bars=True)

# plot the posterior
prior_limits = solver._get_limits()
fig, axes = analysis.pairplot(
    samples.cpu().numpy(),
    limits=prior_limits,
    # ticks=[[], []],
    figsize=(10, 10),
    points=theta.cpu().numpy(),
    points_offdiag={"markersize": 5, "markeredgewidth": 1},
    points_colors="r",
    labels=[],
    upper=["kde"],
    diag=["kde"],
)

# find the mode of the posterior using "kde"
param_values_estimated = []
for i in tqdm(range(len(prior_limits))):
    kde = gaussian_kde(samples[:, i])
    prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 5000)
    densities = kde.evaluate(prior_range)
    param_value = prior_range[np.argmax(densities)]
    param_values_estimated.append(param_value)


# prepare one seqC trained
# load the pretrained model

# prepare one seqC validation


# compute all samples distance
