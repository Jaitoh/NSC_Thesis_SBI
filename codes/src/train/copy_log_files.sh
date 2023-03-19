# copy from server to local
# scp 
# snn server
scp -r wehe@130.60.191.245:tmp/NSC/codes/src/train/logs/ ./src/train/logs/
rsync -azn --progress wehe@130.60.191.245:tmp/NSC/codes/src/train/logs ./src/train/logs_snn2
rsync -az --progress wehe@130.60.191.245:tmp/NSC/codes/src/train/logs ./src/train/logs_snn2

# snn data x, theta
rsync -azn --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_3.pt ../data/training_datasets/
rsync -azn --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_5.pt ../data/training_datasets/
rsync -azn --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_6.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_3.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_5.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_6.pt ../data/training_datasets/

rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/x_15_3.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/x_15_5.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/x_15_6.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/x_test.pt ../data/training_datasets/
rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_test.pt ../data/training_datasets/

# sensors server
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/codes/src/train/logs ./src/train/logs_sensors
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/codes/src/train/logs ./src/train/logs_sensors
# data x, theta
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_1.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_2.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_4.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_5.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_1.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_2.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_4.pt ../data/training_datasets/
rsync -azn --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_5.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_1.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_2.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_4.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/theta_15_5.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_1.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_2.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_4.pt ../data/training_datasets/
rsync -az --progress wenjie@10.65.53.20:~/NSC_Thesis/data/training_datasets/x_15_5.pt ../data/training_datasets/

