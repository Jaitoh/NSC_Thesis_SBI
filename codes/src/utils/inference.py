import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import gaussian_kde

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.setup import adapt_path


# load config
def load_stored_config(exp_dir):
    config_path = Path(exp_dir) / "config.yaml"
    config_path = adapt_path(config_path)
    print(f"==>> config_path: {config_path}")
    config = OmegaConf.load(config_path)
    config.log_dir = str(exp_dir)

    model_path = Path(exp_dir) / "model" / "best_model.pt"
    # check if model exists
    if not model_path.exists():
        model_path = Path(exp_dir) / "model" / "model_check_point.pt"

    return config, model_path


def get_posterior(model_path, config, device, Solver, low_batch=0, return_dataset=False):
    """get the trained posterior"""

    solver = Solver(config, training_mode=False)
    train_loader, valid_loader, network = solver.init_inference(sum_writer=False).prepare_dataset_network(
        config,
        model_path,
        device=device,
        low_batch=low_batch,
    )
    posterior = solver.inference.build_posterior(network)
    solver.inference._model_bank = []
    print(f"finished building posterior")
    if return_dataset:
        return solver, posterior, train_loader, valid_loader
    else:
        return solver, posterior


def estimate_theta_from_post_samples(prior_limits, samples):
    # move samples to cpu if necessary
    if samples.is_cuda:
        samples = samples.cpu().detach().numpy()

    # estimate theta values from samples using KDE
    theta_estimated = []
    for i in tqdm(range(len(prior_limits))):
        kde = gaussian_kde(samples[:, i])
        prior_range = np.linspace(prior_limits[i][0], prior_limits[i][1], 2500)
        densities = kde.evaluate(prior_range)
        theta_value = prior_range[np.argmax(densities)]
        theta_estimated.append(theta_value)

    return theta_estimated


def sampling_from_posterior(device, posterior, x_o, num_samples=20000, show_progress_bars=False):
    return (
        posterior.sample(
            (num_samples,),
            x=x_o.cuda() if device == "cuda" else x_o,
            show_progress_bars=show_progress_bars,
        )
        .cpu()
        .numpy()
    )


def convert_normed_theta(theta_estimated_normed, prior_min, prior_max, prior_labels):
    # convert theta from 0-1 to original range
    original_range_pair = [[x, y] for x, y in zip(prior_min, prior_max)]

    theta_estimated = []
    for i in range(len(theta_estimated_normed)):
        theta_estimated.append(
            theta_estimated_normed[i] * (original_range_pair[i][1] - original_range_pair[i][0])
            + original_range_pair[i][0]
        )

    return theta_estimated, original_range_pair


def convert_samples_range(samples, mapped_limits, original_limits):
    # map samples from mapped_limits to original_limits
    for i in range(len(mapped_limits)):
        mapped_low = mapped_limits[i][0]
        mapped_up = mapped_limits[i][1]
        origin_low = original_limits[i][0]
        origin_up = original_limits[i][1]

        if len(samples.shape) == 1:
            samples[i] = (samples[i] - mapped_low) / (mapped_up - mapped_low) * (
                origin_up - origin_low
            ) + origin_low
        else:
            samples[:, i] = (samples[:, i] - mapped_low) / (mapped_up - mapped_low) * (
                origin_up - origin_low
            ) + origin_low

    return samples
