source activate sbi

exp_id=L0-nle-p2-cnn/L0-nle-p2-cnn-dur3to11-post

exp_dir=~/tmp/NSC/codes/src/train_nle/logs/$exp_id
mkdir -p $exp_dir/inference
PRINT_LOG=$exp_dir/inference/inference.log
code $PRINT_LOG
nice python3 -u ~/tmp/NSC/codes/src/inference/subj_inference_nle_p2.py \
    -e $exp_dir \
    >${PRINT_LOG} 2>&1
