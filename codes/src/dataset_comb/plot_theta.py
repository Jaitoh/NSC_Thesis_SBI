import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./src')
import h5py

def plot_theta_distribution():
    """ plot theta sampling from .h5 file 
    
    """
    # load h5 file
    h5_file = '../data/data_for_sbi_sample100_s1.h5'
    with h5py.File(h5_file, 'r') as f:
        theta = f['theta'][:]
        x = f['x'][:]
        
    
plot_theta_distribution()