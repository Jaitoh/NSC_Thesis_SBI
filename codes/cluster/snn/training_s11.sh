export CUDA_VISIBLE_DEVICES=1
python3 ./src/train/test_train_L0_test.py > ./src/train/logs/log_test.txt
python3 ./src/train/test_train_L0_4.py > ./src/train/logs/log_sample_Rchoices4.txt
python3 ./src/train/test_train_L0_5.py > ./src/train/logs/log_sample_Rchoices5.txt