"""
tmp_save2cpu.py
"""
from pathlib import Path

import torch
import pickle
from sbi import utils as utils
from typing import Any, Callable
import sbi.inference

def check_method(method):
    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError("Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'.")
    return method_fun

log_dir = Path('./src/train/logs_15_p0/log_sample_Rchoices2')
density_estimator_dir = log_dir / 'density_estimator.pkl'
inference_dir = log_dir / 'inference.pkl'
posterior_dir = log_dir / 'posterior.pkl'

with open(density_estimator_dir, 'rb') as f:
    density_estimator = pickle.load(f)

method = 'snpe'
method_fun = check_method(method)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
prior_min_train = [-3.7, 0, 0, -5]
prior_max_train = [2.5, 71, 18, 7]
prior_train = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min_train), high=torch.as_tensor(prior_max_train), device=device
)

inference = method_fun(prior=prior_train,
                       density_estimator='maf',
                       device=device,
                       logging_level='WARNING',
                       # summary_writer=writer,
                       show_progress_bars=True,
                       )
# save the density_estimator to cpu
density_estimator.to('cpu')
posterior = inference.build_posterior(density_estimator)

with open(density_estimator_dir, 'wb') as f:
    pickle.dump(density_estimator, f)
print('density_estimator saved to', density_estimator_dir)
with open(posterior_dir, 'wb') as f:
    pickle.dump(posterior, f)
print('posterior saved to', posterior_dir)
with open(inference_dir, 'wb') as f:
    pickle.dump(inference, f)
print('inference saved to', inference_dir)
