CLUSTER=uzh

DATA_DIR="/home/wehe/scratch/data/feature/v2"
MERGED_DATA_path=/home/wehe/scratch/data/feature-L0-Eset0-98sets-T500v2-C100.h5

PRINT_DIR="./cluster/${CLUSTER}/dataset/process"
PRINT_LOG="./cluster/${CLUSTER}/dataset/process/merge-feature-L0-Eset0-98sets-T500v2-C100.log"

echo "print_log: ${PRINT_LOG}"
code ${PRINT_LOG}
# mkdir -p $DATA_DIR
# mkdir -p $PRINT_DIR

cd /home/wehe/data/NSC/codes
/data/wehe/conda/envs/sbi/bin/python3 -u ./src/features/merge_features.py \
    --data_dir ${DATA_DIR} \
    --merged_data_path ${MERGED_DATA_path} >${PRINT_LOG} 2>&1

echo 'finished dataset merge'

cd /home/wehe/data
./gdrive files upload ${MERGED_DATA_path} >/home/wehe/data/NSC/codes/${PRINT_LOG}.2 2>&1
