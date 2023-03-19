export CUDA_VISIBLE_DEVICES=0
python3 ./src/train/test_train_L0_test.py > ./src/train/logs/log_test.txt
python3 ./src/train/test_train_L0_1.py > ./src/train/logs/log_sample_Rchoices1.txt
python3 ./src/train/test_train_L0_2.py > ./src/train/logs/log_sample_Rchoices2.txt
python3 ./src/train/test_train_L0_3.py > ./src/train/logs/log_sample_Rchoices3.txt
