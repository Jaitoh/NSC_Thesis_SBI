# Simulation-based Bayesian Inference for A Stochastic Decision Making Model

<!-- ./gdrive files upload --recursive <FILE_PATH> -->

unzip files for example:
```
tar -xzf codes.tar.gz
```


## Results reproduction
The results or the plots can be reproduced from the following notebooks:
- `./codes/notebooks/01. NLE.ipynb`
- `./codes/notebooks/02. NPE.ipynb`
- `./codes/notebooks/03. FNPE.ipynb`
- `./codes/notebooks/04. inferences.ipynb`

In addition, some useful plots can be found in `./codes/notebooks/05. prior.ipynb`

## Environment setup
navigate to the root folder where you can see 
- codes
- data
- readme.md
- setup.sh

On a Linux machine, with conda installed, run the following commands to create the environment and install the required packages:

``` bash
./setup.sh
```

unfold some log files
```bash
tar -zxf ./codes/src/train/logs.tar.gz -C ./codes/src/train
tar -zxf ./codes/src/train_nle/logs.tar.gz -C ./codes/src/train_nle
tar -zxf ./codes/notebook/figures.tar.gz -C ./codes/notebook/

mv ./codes/src/train/home/wehe/data/NSC_submit/codes/src/train/logs ./codes/src/train
mv ./codes/src/train_nle/home/wehe/data/NSC_submit/codes/src/train_nle/logs ./codes/src/train_nle
mv ./codes/notebook/home/wehe/data/NSC_submit/codes/notebook/figures ./codes/notebook


rm -r ./codes/src/train/home
rm -r ./codes/src/train_nle/home
rm -r ./codes/notebook/home

```

## Build the simulator
The cython simulator should be build before run the simulations whenever the simulator is modified or the machine is changed.

1. navigate to `./codes/src/simulator`
2. run `./DM_compute_build.sh`
3. then navigate back to the root folder

```bash
cd ./codes/src/simulator
./DM_compute_build.sh
cd ../../../
```

## Download data required for the experiments
make sure the following simulated dataset are in the `./data/dataset` folder
- `dataset-L0-Eset0-100sets-T500.h5` [1.18GB]
- `feature-L0-Eset0-100sets-T500-C100` [2.59GB]


## train NPE
```bash
./codes/src/train/do_train_p5a.sh
```
before running the bash script, you might need to change the ROOT_DIR in the script to the correct path.

## train FNPE
```bash
./codes/src/train/do_train_p4a.sh
```
before running the bash script, you might need to change the ROOT_DIR in the script to the correct path.

## train NLE
```bash
./codes/src/train_nle/do_train_nle-p3.sh
```
before running the bash script, you might need to change the ROOT_DIR in the script to the correct path.

inference with NLE using test data
```bash
./codes/notebook/nle_inference.sh
```

## Inference using the DM-Model
The inference time using the DM-Model can be find from the file `./codes/notebook/DM_inference.log`
To reproduce the inference, run the following command:
```bash
./codes/notebook/DM_inference.sh
```