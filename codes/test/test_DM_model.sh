#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 ./test/test_DM_model.py
python3 ./test/test_DM_model_speed_test.py