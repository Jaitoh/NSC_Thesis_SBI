# log p4-4Fs-1D-cnn.tar.gz
TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-online-copy
FILE_ID=1vukn9tNDBJPHEBakCcHr2i-NYTJVaftA

# LOG_DIR="/home/wehe/tmp/NSC/codes/src/train_nle/logs"
LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train_nle/logs"
mkdir ${LOG_DIR}/${TRAIN_ID}

cd ~
./gdrive files download ${FILE_ID} \
    --destination ${LOG_DIR}/${TRAIN_ID}

cd ${LOG_DIR}/${TRAIN_ID}
tar -xzf ${EXP_ID}.tar.gz

# move to current directory
LOG_DIR="/home/wehe/tmp/NSC/codes/src/train_nle/logs"
LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train_nle/logs"
mv ./${LOG_DIR}/${TRAIN_ID}/${EXP_ID} ./${EXP_ID}
rm -r ./home
rm ${EXP_ID}.tar.gz
echo "Done"
