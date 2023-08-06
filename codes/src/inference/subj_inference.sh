exp_id=train_L0_p5a/p5a-conv_net
exp_dir=~/tmp/NSC/codes/src/train/logs/$exp_id

PRINT_LOG=$exp_dir/inference/inference.log

nice python3 -u ~/tmp/NSC/codes/src/inference/subj_inference.py \
    -e $exp_dir \
    >${PRINT_LOG} 2>&1
