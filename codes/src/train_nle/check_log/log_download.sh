# log p4-4Fs-1D-cnn.tar.gz
TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-online-copy
FILE_ID=1vukn9tNDBJPHEBakCcHr2i-NYTJVaftA

TRAIN_ID=L0-nle-cnn
EXP_ID=L0-nle-cnn-dur3-online
FILE_ID=1vX1rN9nDP8-hi2kVkXfYHgJOH-RhhI1z

# TRAIN_ID=L0-nle-cnn
# EXP_ID=L0-nle-cnn-dur3-offline
# FILE_ID=1X2JJLyS8DEgA7b4mrloWW_J2a2AXVxyp

# TRAIN_ID=L0-nle-cnn
# EXP_ID=L0-nle-cnn-dur3-offline_acc
# FILE_ID=1jRg62XFF3OcPzIIYs-DbuXzrawriY50y

LOG_DIR="$HOME/tmp/NSC/codes/src/train_nle/logs"
mkdir ${LOG_DIR}/${TRAIN_ID}

cd ~
./gdrive files download ${FILE_ID} \
    --destination ${LOG_DIR}/${TRAIN_ID}

cd ${LOG_DIR}/${TRAIN_ID}
tar -xzf ${EXP_ID}.tar.gz

echo "finished tar"

# move to current directory
MACHINE_LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train_nle/logs" # !
mv .${MACHINE_LOG_DIR}/${TRAIN_ID}/${EXP_ID} ./${EXP_ID}
rm -r ./home
rm ${EXP_ID}.tar.gz
echo "Done"
