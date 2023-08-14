source activate sbi

exp_id=train_L0_p5a/p5a-conv_net
exp_id=train_L0_p5a/p5a-conv_net-Tv2
exp_id=train_L0_p5a/p5a-conv_lstm-tmp

pipeline=p4
exp_id=train_L0_p4/p4-5Fs-1D-cnn

exp_dir=~/tmp/NSC/codes/src/train/logs/$exp_id
mkdir -p $exp_dir/inference
PRINT_LOG=$exp_dir/inference/inference.log

code $PRINT_LOG

nice python3 -u ~/tmp/NSC/codes/src/inference/subj_inference.py \
    -e $exp_dir \
    -p $pipeline \
    >${PRINT_LOG} 2>&1
