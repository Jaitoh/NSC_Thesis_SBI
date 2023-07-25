DO_ZIP=1
DO_Gdrive=1

TRAIN_ID=L0-nle-cnn
folder_id=186igxRFGFbwz23Z3KAu_EQr-rl3Syz0g

EXP_IDS=(
    "L0-nle-cnn-dur3-online-copy" # 1vukn9tNDBJPHEBakCcHr2i-NYTJVaftA
)

# zip log files
# LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs"
LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train_nle/logs"
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
        # ./gdrive files upload "${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz" | tee /dev/fd/2 | grep 'Id:' | awk '{print $2}'
        file_id=$(./gdrive files upload "${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz" | tee /dev/fd/2 | grep 'Id:' | awk '{print $2}')

        # move to folder train_nle/logs
        ./gdrive files move ${file_id} ${folder_id}

        # remove file
        rm "${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz"
        echo "finished and removed file ${LOG_DIR}/${TRAIN_ID}/${EXP_ID}.tar.gz"
        echo ""
    done
fi
