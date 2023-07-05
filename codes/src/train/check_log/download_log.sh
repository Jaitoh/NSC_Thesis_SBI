# log p4-4Fs-1D-cnn.tar.gz
TRAIN_ID=train_L0_p4
EXP_ID=p4-4Fs-1D-cnn

./gdrive files download 1nYVIZIU8ngv-cVUNNsdYQ4RgeFms0SHl \
    --destination /home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_ID}

cd /home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_ID}
tar -xzf ${EXP_ID}.tar.gz
mv ./home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_ID}/${EXP_ID} ./${EXP_ID}
rm -r ./home
rm ${EXP_ID}.tar.gz
