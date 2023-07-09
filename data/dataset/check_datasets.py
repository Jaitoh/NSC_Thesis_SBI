import h5py
from pprint import pprint

dataset_path = "/home/ubuntu/tmp/NSC/data/dataset/dataset-L0-Eset0-98sets-T500v2.h5"

with h5py.File(dataset_path, "r") as f:
    pprint(f.keys())
    pprint(len(f.keys()))
