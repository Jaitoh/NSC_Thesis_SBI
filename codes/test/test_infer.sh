#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 ./test/test_infer.py > ./test/test_infer.txt