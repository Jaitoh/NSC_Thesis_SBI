# copy from server to local
# scp 
# snn server
scp -r wehe@130.60.191.245:tmp/NSC/codes/src/train/logs/ ./src/train/logs/
rsync -azn --progress wehe@130.60.191.245:tmp/NSC/codes/src/train/logs ./src/train/logs_snn
rsync -az --progress wehe@130.60.191.245:tmp/NSC/codes/src/train/logs ./src/train/logs_snn
rsync -az --progress wehe@130.60.191.245:tmp/NSC/codes/cluster/train_L0/ ./cluster/train_L0_snn

# snn data x, theta
rsync -azn --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/theta_15_3.pt ../data/training_datasets/

rsync -az --progress wehe@130.60.191.245:~/tmp/NSC/data/training_datasets/x_15_3.pt ../data/training_datasets/

# sensors server
rsync -azn --progress wenjie@10.65.53.20:~/NSC/codes/src/train/logs ./src/train/logs_sensors
rsync -az --progress wenjie@10.65.53.20:~/NSC/codes/src/train/logs ./src/train/logs_sensors

# from uzh cluster
rsync -az --progress wehe@cluster.s3it.uzh.ch:~/data/NSC/codes/cluster/uzh/train_L0_v1 ./cluster/uzh/train_L0_uzh
rsync -azn --progress wehe@cluster.s3it.uzh.ch:~/data/NSC/codes/src/train/logs/ ./src/train/logs_uzh
