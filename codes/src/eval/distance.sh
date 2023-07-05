PYTHONPATH="home/ubuntu/tmp/NSC/codes:${PYTHONPATH}"
pwd

TRAIN_ID="train_L0_p4"
EXP_IDS=(
    "p4-4Fs-1D-cnn2-0"
    "p4-4Fs-1D-cnn"
    "p4-3Fs-1D-cnn"
    "p4-4Fs-1D-cnn2-size2"
    "p4-5Fs-1D-cnn"
    "p4-5Fs-2D-mh_cnn"
)

ROOT_DIR="/home/ubuntu/tmp/NSC/codes"
LOG_DIR="${ROOT_DIR}/src/train/logs/" #${TRAIN_ID}/${EXP_ID}

for EXP_ID in "${EXP_IDS[@]}"; do

    OUT_DIR="${ROOT_DIR}/src/eval/${TRAIN_ID}/${EXP_ID}"
    mkdir -p ${OUT_DIR}
    echo "OUT_DIR: ${OUT_DIR}"
    nice python3 ./src/eval/distance_p4.py \
        hydra.run.dir=${ROOT_DIR} \
        OUT_DIR=${OUT_DIR} \
        LOG_DIR=${LOG_DIR} \
        EXP_ID="${TRAIN_ID}/${EXP_ID}" \
        num_C=100 \
        num_estimation=3 \
        >${OUT_DIR}/distance.log 2>&1 #&
    code ${OUT_DIR}/distance.log
done

#=================================================================================================== p4 t-2
# TRAIN_ID="train_L0_p4"
# EXP_IDS=(
#     "p4-3Fs-1D-mlp"
#     "p4-5Fs-1D-mlp"
#     "p4-F2-1D-cnn"
#     "p4-F3-1D-mlp"
#     "p4-F5-1D-cnn"
#     "p4-4Fs-1D-cnn2-size1"
#     "p4-F1-1D-cnn"
#     "p4-F2-1D-mlp"
#     "p4-F4-1D-cnn"
#     "p4-F5-1D-mlp"
#     "p4-4Fs-1D-mlp"
#     "p4-F1-1D-mlp"
#     "p4-F3-1D-cnn"
#     "p4-F4-1D-mlp"
# )

# ROOT_DIR="/home/ubuntu/tmp/NSC/codes"
# LOG_DIR="${ROOT_DIR}/src/train/logs/" #${TRAIN_ID}/${EXP_ID}

# for EXP_ID in "${EXP_IDS[@]}"; do
# OUT_DIR="${ROOT_DIR}/src/eval/${TRAIN_ID}/${EXP_ID}"

# mkdir -p ${OUT_DIR}
#     nice python3 ./src/eval/distance_p4.py \
#         hydra.run.dir=${ROOT_DIR} \
#         OUT_DIR=${OUT_DIR} \
#         LOG_DIR=${LOG_DIR} \
#         EXP_ID="${TRAIN_ID}/${EXP_ID}" \
#         num_C=100 \
#         num_estimation=3 \
#         >${OUT_DIR}/distance.log 2>&1 &
#     code ${OUT_DIR}/distance.log
# done

#
# TRAIN_ID="train_L0_p5"
# EXP_IDS=(
#     "p5-conv_transformer"
#     "p5-gru3"
# )

# ROOT_DIR="/home/ubuntu/tmp/NSC/codes"
# LOG_DIR="${ROOT_DIR}/src/train/logs/" #${TRAIN_ID}/${EXP_ID}

# for EXP_ID in "${EXP_IDS[@]}"; do
# OUT_DIR="${ROOT_DIR}/src/eval/${TRAIN_ID}/${EXP_ID}"

# mkdir -p ${OUT_DIR}
#     nice python3 ./src/eval/distance_p4.py \
#         hydra.run.dir=${ROOT_DIR} \
#         OUT_DIR=${OUT_DIR} \
#         LOG_DIR=${LOG_DIR} \
#         EXP_ID="${TRAIN_ID}/${EXP_ID}" \
#         num_C=100 \
#         num_estimation=3 \
#         >${OUT_DIR}/distance.log 2>&1 &
#     code ${OUT_DIR}/distance.log
# done

# #
# TRAIN_ID="train_L0_p5"
# EXP_IDS=(
#     "p5-conv_lstm"
# )

# ROOT_DIR="/home/ubuntu/tmp/NSC/codes"
# LOG_DIR="${ROOT_DIR}/src/train/logs/" #${TRAIN_ID}/${EXP_ID}

# for EXP_ID in "${EXP_IDS[@]}"; do
# OUT_DIR="${ROOT_DIR}/src/eval/${TRAIN_ID}/${EXP_ID}"

# mkdir -p ${OUT_DIR}
#     nice python3 ./src/eval/distance_p4.py \
#         hydra.run.dir=${ROOT_DIR} \
#         OUT_DIR=${OUT_DIR} \
#         LOG_DIR=${LOG_DIR} \
#         EXP_ID="${TRAIN_ID}/${EXP_ID}" \
#         num_C=100 \
#         num_estimation=3 \
#         >${OUT_DIR}/distance.log 2>&1 &
# code ${OUT_DIR}/distance.log
# done
