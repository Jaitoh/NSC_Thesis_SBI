DO_ZIP=1
DO_Gdrive=1

# === npe p5a ===
Pipeline=train # npe
TRAIN_ID=train_L0_p5a
folder_id=1wWIw8E3k_EgJ3F1gfHNqOG7CEu04MYkj

# "p5a-conv_net"
# "p5a-conv_lstm-corr_conv-tmp-2"
# "p5a-conv_lstm-maf3-tmp"
# "p5a-conv_lstm-corr_conv-tmp-4"
# "p5a-conv_lstm-maf3-tmp-2"
EXP_IDS=(
    "p5a-conv_lstm-corr_conv"
)
# "p5a-conv_lstm-Tv2"
# "p5a-conv_net-Tv2-tmp"

# Pipeline=train # npe
TRAIN_ID=train_L0_p4a
folder_id=1SiXpFibwBr9OXA6Oi7tdj4PaX_UcgCbg

# "p4a-F1345-cnn-tmp"
# "p4a-F134-cnn"
# "p4a-F1345-cnn"
# "p4a-F134-cnn-maf3-tmp"
# "p4a-F1345-cnn-maf3-tmp"
EXP_IDS=(
    "p4a-F1345-cnn-maf3"
    "p4a-F134-cnn-maf3"
)

# === nle ===
# Pipeline=train_nle
# TRAIN_ID=L0-nle-cnn
# folder_id=186igxRFGFbwz23Z3KAu_EQr-rl3Syz0g # logs gdrive folder id

# "L0-nle-cnn-dur3-online"
# "L0-nle-cnn-dur3-offline"
# "L0-nle-cnn-dur7-offline_acc"
# EXP_IDS=(
#     "p5a-conv_net-old_net"
#     "p5a-conv_net-Tv2-old_net"
# )

# === nle p2 ===
Pipeline=train_nle
TRAIN_ID=L0-nle-p2-cnn
folder_id=1wJE5wfoMCi-hUZAGNsiIE4QLFsVCIqsu # logs gdrive folder id

# "L0-nle-p2-cnn-dur3"
# "L0-nle-p2-cnn-dur3to7"
# "L0-nle-p2-cnn-dur3to11-post"
# "L0-nle-p2-cnn-datav2-small-batch-tmp"
# "L0-nle-p2-cnn-datav2-tmp"
EXP_IDS=(
    "L0-nle-p2-cnn-datav2"
    "L0-nle-p2-cnn-datav2-small-batch"
)

# === nle p3 ===
# Pipeline=train_nle
# TRAIN_ID=L0-nle-p3-cnn
# folder_id=1ySMYmMdK7D0Jyx8TF5Bq4Iy4qGt32j93 # logs gdrive folder id

# EXP_IDS=(
#     "L0-nle-p3-cnn-tmp"
# )

# zip log files
# LOG_DIR="/home/ubuntu/tmp/NSC/codes/src/train/logs"
LOG_DIR=~/tmp/NSC/codes/src/${Pipeline}/logs
if [ ${DO_ZIP} -eq 1 ]; then
    for EXP_ID in "${EXP_IDS[@]}"; do
        cd ${LOG_DIR}/${TRAIN_ID}
        EXP_DIR="${LOG_DIR}/${TRAIN_ID}/${EXP_ID}"
        # if EXP_DIR does not exist, stop and print error message
        if [ ! -d ${EXP_DIR} ]; then
            echo "Error: ${EXP_DIR} does not exist"
            exit 1
        fi
        # cp ./${EXP_ID}.log ${EXP_DIR}/
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
