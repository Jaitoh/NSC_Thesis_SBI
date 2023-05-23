# torch
conda create -n sbi python=3.8 -y
source activate sbi
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# pip install torch torchvision torchaudio pytorch-cuda #--index-url https://download.pytorch.org/whl/cu118
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install sbi
pip install opencv-python
pip install jupyter h5py numpy pandas cython snakeviz
pip install tqdm matplotlib tensorboard
pip install gpustat pyyaml memory_profiler
pip install imageio
pip install torch_tb_profiler
# pip install tensorboard
# pip install memory_profiler