# %%
"""
generate plots for training logs
- training curves
"""
# magic command to reload modules
# %load_ext autoreload
# %autoreload 2

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import sys
import yaml
from sbi import analysis
import seaborn as sns

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")
from utils.setup import adapt_path
from utils.event import get_train_valid_lr
from utils.plots import load_img, pairplot, plot_posterior_mapped_samples
from utils.inference import (
    get_posterior,
    load_stored_config,
    sampling_from_posterior,
    convert_samples_range,
)
from utils.train import WarmupScheduler, plot_posterior_with_label, load_net, get_limits

import matplotlib as mpl
import matplotlib.pyplot as plt

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
# remove all edges
mpl.rcParams["axes.edgecolor"] = "k"
mpl.rcParams["axes.linewidth"] = 2
font = {
    "weight": "bold",
    "size": 22,
}
mpl.rc("font", **font)

# %% ==================================================
pipeline_version = "p5a"
train_id = "train_L0_p5a"
exp_id = "p5a-conv_lstm-corr_conv-tmp"

# %% load the latest event file
log_dir = Path(NSC_DIR) / "codes/src/train/logs" / train_id / exp_id
(
    wall_time,
    step_nums,
    learning_rates,
    log_probs_train,
    log_probs_valid,
) = get_train_valid_lr(log_dir)

all_probs = np.concatenate([log_probs_train, log_probs_valid])
upper = np.max(all_probs)
lower = np.percentile(all_probs, 10)

# %% load the config.yaml file
config_file = log_dir / "config.yaml"
config = adapt_path(config_file)

with open(config, "r") as f:
    config = yaml.safe_load(f)

prior_min = config["prior"]["prior_min"]
prior_max = config["prior"]["prior_max"]

# %% plot training curves
fig, ax = plt.subplots(figsize=(21, 9))
ax.plot(step_nums, log_probs_train, label="train", alpha=0.6, ms=0.2, color="tab:gray")
ax.plot(step_nums, log_probs_valid, label="validation", alpha=0.9, ms=0.2, color="k")
ax.set_ylim(lower, upper)
ax.set_xlabel("epoch")
ax.set_ylabel("$\log(p)+\log|det|$")
ax.grid(alpha=0.2)
ax.legend()

ax1 = ax.twiny()
ax1.plot(
    (np.array(wall_time) - wall_time[0]) / 60 / 60,
    max(log_probs_valid) * np.ones_like(log_probs_valid),
    "-",
    alpha=0,
)
ax1.set_xlabel("time (hours)")


# %% plot posterior training process
num_epoch = 6
best_epoch = step_nums[np.argmax(log_probs_valid)]

# get recorded posterior plot names
img_folder = Path(f"{log_dir}/posterior/figures")
posterior_plots = img_folder.glob("posterior_seen_0*.png")
posterior_idx = [eval(str(plot).split("epoch_")[-1].split(".png")[0]) for plot in posterior_plots]
posterior_idx = np.array(posterior_idx)[np.argsort(posterior_idx)]

chosen_idx = np.linspace(10, len(posterior_idx) - 1, num_epoch, dtype=int)

fig, axes = plt.subplots(1, num_epoch, figsize=(num_epoch * 4, 4))

BL_coor = [731, 731 + 170, 99, 99 + 173]  # x_start, x_end, y_start, y_end
BL_labels = ["$\lambda$", "bias"]
AS_coor = [528, 528 + 170, 304, 304 + 173]  # x_start, x_end, y_start, y_end
AS_labels = ["$\sigma^2_s$", "$\sigma^2_a$"]

coor = BL_coor
labels = BL_labels

# coor = AS_coor
# labels = AS_labels

for i, epoch_idx in enumerate(chosen_idx):
    ax = axes[i]
    img_path = img_folder / f"posterior_seen_0_epoch_{posterior_idx[epoch_idx]}.png"
    if img_path.exists():
        load_img(
            img_path=img_path,
            ax=ax,
            title=f"epoch {posterior_idx[epoch_idx]}",
            crop=True,
            x_start=coor[0],
            x_end=coor[1],
            y_start=coor[2],
            y_end=coor[3],
        )
    ax.set_xlabel(labels[0])
    if i == 0:
        ax.set_ylabel(labels[1])


# %%
config, model_path = load_stored_config(exp_dir=log_dir)

if "p4" in pipeline_version:
    from train.train_L0_p4a import Solver
if "p5" in pipeline_version:
    from train.train_L0_p5a import Solver

solver, posterior, train_loader, valid_loader = get_posterior(
    model_path=model_path,
    config=config,
    device="cuda",
    Solver=Solver,
    # low_batch=20,
    return_dataset=True,
)

# load one sample
seen_data = solver.inference.seen_data_for_posterior
xy_o, true_theta = seen_data["x"][0], seen_data["theta"][0]

# xy_o, true_theta = next(train_loader.__iter__())
# xy_o, true_theta = xy_o[0], true_theta[0]

# %%
normed_limits = solver._get_limits()
final_limits = get_limits(config.prior.prior_min, config.prior.prior_max)

font = {
    "weight": "bold",
    "size": 12,
}
mpl.rc("font", **font)
mpl.rcParams["axes.linewidth"] = 1

plot_posterior_mapped_samples(
    posterior,
    xy_o,
    true_theta=true_theta,
    num_samples=20_000,
    sampling_device="cuda",
    show_progress_bars=False,
    original_limits=normed_limits,
    mapped_limits=final_limits,
)

# %% change one of the parameter and see how the prediction changes
# generate dataset for this testing case
x_o = xy_o[:, :-1]
print(f"==>> x_o.shape: {x_o.shape}")

print(f"{final_limits=}")
ref_theta_values = [1, 1, 1, 1]


# %% evaluate the performance of the trained model
# on whole trained dataset
# on whole validation dataset
