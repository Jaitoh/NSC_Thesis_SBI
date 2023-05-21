# torch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pyyaml
# memory_profiler
# jupyter
# h5py
# numpy
# pandas
# cython
# tqdm
# matplotlib
# gpustat
# opencv-python
# snakeviz
conda install -y -c anaconda jupyter h5py numpy pandas cython snakeviz 
conda install -y -c conda-forge tqdm matplotlib 
conda install -y -c conda-forge gpustat
conda install -y -c conda-forge tensorboard 
conda install -y -c conda-forge pyyaml
conda install -y -c conda-forge ffmpeg
conda install -y -c conda-forge memory_profiler
pip install opencv-python
pip install sbi