import torch
import os
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def seed_worker(worker_id):
#     # Set a fixed seed for the worker
#     np.random.seed(int(42 + worker_id))
#     random.seed(int(42 + worker_id))
