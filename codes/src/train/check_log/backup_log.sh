DO_ZIP=1
DO_Gdrive=1

TRAIN_ID=train_L0_p4a
TRAIN_ID=train_L0_p5a

# "p4-5Fs-1D-cnn"
# "p4-4Fs-1D-cnn"
# "p4-3Fs-1D-cnn"
# "p4-5Fs-2D-mh_cnn"
# "p4-F1-1D-cnn"
# "p4-F2-1D-cnn"
# "p4-F3-1D-cnn"
# "p4-F4-1D-cnn"
# "p4-F5-1D-cnn"
# "p4-5Fs-1D-mlp"
# "p4-4Fs-1D-mlp"
# "p4-3Fs-1D-mlp"
# "p4-F1-1D-mlp"
# "p4-F2-1D-mlp"
# "p4-F3-1D-mlp"
# "p4-F4-1D-mlp"
# "p4-F5-1D-mlp"
# "p4-4Fs-1D-cnn2-0"
# "p4-4Fs-1D-cnn2-size2"
# "p5-gru3"
# "p5-conv_transformer"
# "p5-conv_lstm"
# "p4-4Fs-1D-cnn2-size1"
# "p4a-4Fs-cnn-maf-Tnorm-datasize1"
# "p5a-conv_lstm-ignore_ss"
# "p4a-4Fs-cnn-ignore_ss"
# "p4a-4Fs-cnn-Tnorm"
# "p4a-4Fs-cnn-Tv2-0"
# "p4a-4Fs-cnn-Tv2-1"
EXP_IDS=(
    "p5a-conv_lstm-Tv2-0"
    "p5a-conv_lstm-Tv2-0-ctd"
    "p5a-conv_lstm-Tv2-1"
    "p5a-conv_lstm-Tv2-1-ctd"
)

# zip log files
LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs"
if [ ${DO_ZIP} -eq 1 ]; then
    for EXP_ID in "${EXP_IDS[@]}"; do
        cd ${LOG_DIR}/${TRAIN_ID}
        EXP_DIR="${LOG_DIR}/${TRAIN_ID}/${EXP_ID}"
        # if EXP_DIR does not exist, stop and print error message
        if [ ! -d ${EXP_DIR} ]; then
            echo "Error: ${EXP_DIR} does not exist"
            exit 1
        fi
        tar -zcf ${EXP_DIR}.tar.gz ${EXP_DIR}
    done
fi

# upload to google drive
if [ ${DO_Gdrive} -eq 1 ]; then
    # upload to google drive
    for EXP_ID in "${EXP_IDS[@]}"; do
        cd ~
        ./gdrive files upload "${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz"
        rm "${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz"
        echo "finished and removed file ${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz"
        echo ""
    done
fi
