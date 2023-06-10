#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 ./test/cProfile_train_L0.py
snakeviz output.dat