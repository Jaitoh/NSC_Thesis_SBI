# log p4-4Fs-1D-cnn.tar.gz
TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-online-copy
FILE_ID=1vukn9tNDBJPHEBakCcHr2i-NYTJVaftA

TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-online
FILE_ID=1vX1rN9nDP8-hi2kVkXfYHgJOH-RhhI1z

TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-offline
FILE_ID=1X2JJLyS8DEgA7b4mrloWW_J2a2AXVxyp

# LOG_DIR="/home/wehe/tmp/NSC/codes/src/train_nle/logs"
LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train_nle/logs"

./gdrive files download ${FILE_ID} \
    --destination ${LOG_DIR}/${TRAIN_ID}

cd ${LOG_DIR}/${TRAIN_ID}
tar -xzf ${EXP_ID}.tar.gz
mv ${LOG_DIR}/${TRAIN_ID}/${EXP_ID} ./${EXP_ID}
rm -r ./home
rm ${EXP_ID}.tar.gz
