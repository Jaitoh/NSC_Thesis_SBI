TRAIN_ID=train_L0_p4

# "p4-5Fs-1D-cnn"
# "p4-4Fs-1D-cnn"
# "p4-3Fs-1D-cnn"
# "p4-5Fs-2D-mh_cnn"
EXP_IDS=(
)
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
# "p4-4Fs-cnn2-0"
# "p4-4Fs-cnn2-size1"
# "p4-4Fs-cnn2-size2"
# "p5-gru3"
# "p5-conv_lstm"
# "p5-conv_transformer"

cd ../logs/${TRAIN_ID}

for EXP_ID in "${EXP_IDS[@]}"; do
    FOLDER="/home/ubuntu/tmp/NSC/codes/src/train/logs/${TRAIN_ID}/${EXP_ID}"
    # if folder does not exist, stop and print error message
    if [ ! -d ${FOLDER} ]; then
        echo "Error: ${FOLDER} does not exist"
        exit 1
    fi
    tar -zcvf ${EXP_ID}.tar.gz ${FOLDER}
done
