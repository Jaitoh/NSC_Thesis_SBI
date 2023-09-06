source activate sbi

cd ~/data/NSC/codes/

pipeline=p5a
# exp_id=train_L0_p5a/p5a-conv_net
# exp_id=train_L0_p5a/p5a-conv_net-Tv2
exp_id=train_L0_p5a/p5a-conv_lstm-corr_conv-tmp-4

pipeline=p4a
# # exp_id=train_L0_p4/p4-5Fs-1D-cnn
exp_id=train_L0_p4a/p4a-F1345-cnn-maf3

exp_dir=~/data/NSC/codes/src/train/logs/$exp_id
mkdir -p $exp_dir/inference
PRINT_LOG=$exp_dir/inference/inference.log

code $PRINT_LOG

nice python3 -u ~/data/NSC/codes/src/inference/subj_inference.py \
    -e $exp_dir \
    -p $pipeline \
    >${PRINT_LOG} 2>&1
