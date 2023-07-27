# torch
# install python 3.8
conda activate base
conda remove --name sbi --all -y
conda create -n sbi python=3.10 -y
# source activate sbi
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# pip install torch torchvision torchaudio pytorch-cuda #--index-url https://download.pytorch.org/whl/cu118
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda -c pytorch -c nvidia -y
# conda install pytorch torchvision torchaudio pytorch-cuda==11.8 -c pytorch -c nvidia
conda activate sbi
pip install sbi
pip install sbibm
pip install opencv-python pyyaml
pip install jupyter h5py numpy pandas cython snakeviz
pip install tqdm matplotlib tensorboard
pip install gpustat pyyaml memory_profiler
pip install imageio
pip install torch_tb_profiler
pip install hydra-core --upgrade
pip install spyder-kernels
pip install tables
conda activate sbi
####
conda clean --all
pip cache purge

/home/wehe/data/conda/envs/sbi/bin/pip install sbi
/home/wehe/data/conda/envs/sbi/bin/pip install opencv-python pyyaml
/home/wehe/data/conda/envs/sbi/bin/pip install jupyter h5py numpy pandas cython snakeviz
/home/wehe/data/conda/envs/sbi/bin/pip install tqdm matplotlib tensorboard
/home/wehe/data/conda/envs/sbi/bin/pip install gpustat pyyaml memory_profiler
/home/wehe/data/conda/envs/sbi/bin/pip install imageio
/home/wehe/data/conda/envs/sbi/bin/pip install torch_tb_profiler
/home/wehe/data/conda/envs/sbi/bin/pip install hydra-core --upgrade
/home/wehe/data/conda/envs/sbi/bin/pip install spyder-kernels

# pip install tensorboard
# pip install memory_profiler

# module load anaconda3
# source activate sbi
