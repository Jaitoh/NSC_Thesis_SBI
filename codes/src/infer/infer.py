import pickle
from pathlib import Path
from sbi import analysis

# load posterior.pkl
log_dir = Path('./src/train/logs_snn/log_sample_Rchoices')
posterior_dir = log_dir / 'posterior.pkl'
with open(posterior_dir, 'rb') as f:
    posterior = pickle.load(f)

# generate a new data which is not trained

# ...

samples = posterior.sample((1000,), x=x_o)

fig, axes = analysis.pairplot(
    samples,
    limits=[[], []],
    ticks=[[], []],
    figsize=(10, 10),
    points=true_params,
    points_offdiag={'markersize': 10, 'markeredgewidth': 1},
    points_colors='r',
    )
