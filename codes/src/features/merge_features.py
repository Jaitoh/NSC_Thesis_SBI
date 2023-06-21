"""
extracte features from / make summarization of 
seqC and cR

"""
import h5py
import os

# feature_dir = "/home/wehe/tmp/NSC/data/dataset/"
feature_dir = "/home/wehe/scratch/data/feature"

# list all files in the feature_dir starting with "feature"
feature_files = [
    os.path.join(feature_dir, f)
    for f in os.listdir(feature_dir)
    if os.path.isfile(os.path.join(feature_dir, f)) and f.startswith("feature")
]

dest_file = os.path.join(feature_dir, "feature-L0-Eset0-100sets-T500-C100.h5")

for i in range(len(feature_files)):
    print(f"copying {feature_files[i]} [{i}/{len(feature_files)-1}] ... ", end="")
    f = h5py.File(feature_files[i], "r")
    f_dest = h5py.File(dest_file, "a")

    group_name = list(f.keys())[0]
    f_dest.copy(f[group_name], group_name)
    f.close()
    print(f"done.")

f_dest.close()
