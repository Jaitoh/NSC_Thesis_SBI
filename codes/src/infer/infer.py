# infer the posterior from the training and generate plots
from pathlib import Path
import sys
import pickle
sys.path.append('./src')

import torch
from torch.utils.tensorboard import SummaryWriter
import sbi.inference
from sbi import utils as utils
from typing import Any, Callable


# load pkl from file
log_dir = Path('./src/train/logs/log_sample_Rchoices')
density_estimator_dir = log_dir / 'density_estimator.pkl'
with open(density_estimator_dir, 'rb') as f:
    density_estimator = pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device: {device}')

# change from bsss to bss, ignore all 0s
writer = SummaryWriter(log_dir=str(log_dir))
prior_min = [-3.7, -36, 0, -34, 5]
prior_max = [2.5, 71, 0, 18, 7]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max), device=device
)

method = 'snpe'
try:
    method_fun: Callable = getattr(sbi.inference, method.upper())
except AttributeError:
    raise NameError("Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'.")


inference = method_fun(prior=prior,
                       density_estimator='maf',
                       device=device,
                       logging_level='WARNING',
                       summary_writer=writer,
                       show_progress_bars=True,
                       )

posterior = inference.build_posterior(
    density_estimator,
    )

