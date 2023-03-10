import pickle

density_estimator_dir = './src/train/logs/log_sample_Rchoices/density_estimator.pkl'
postior_dir = './src/train/logs/log_sample_Rchoices/posterior.pkl'

with open(density_estimator_dir, 'rb') as f:
    density_estimator = pickle.load(f)

with open(postior_dir, 'rb') as f:
    posterior = pickle.load(f)

# save the posterior to cpu
posterior.to('cpu')
with open(postior_dir, 'wb') as f:
    pickle.dump(posterior, f)

# save the density_estimator to cpu
density_estimator.to('cpu')
with open(density_estimator_dir, 'wb') as f:
    pickle.dump(density_estimator, f)
