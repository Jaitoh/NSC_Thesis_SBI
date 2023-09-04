source activate sbi

export CUDA_VISIBLE_DEVICES=0
cd ~/data/NSC/codes/

subj_ID=$1
batch_theta=10

pipeline_version="nle-p3"
train_id="L0-nle-p3-cnn"
exp_id="L0-nle-p3-cnn-newLoss"
log_exp_id="L0-nle-p3-cnn-newLoss"

exp_dir=~/data/NSC/codes/src/train_nle/logs/$train_id/$exp_id

PRINT_LOG=$exp_dir/posterior/subject_inference_$subj_ID.log
code $PRINT_LOG

nice python3 -u ~/data/NSC/codes/src/inference/subj_inference_nle_p3.py \
    --subj_ID $subj_ID \
    --pipeline_version $pipeline_version \
    --train_id $train_id \
    --exp_id $exp_dir \
    --log_exp_id $log_exp_id \
    --iid_batch_size_theta $batch_theta \
    --num_samples 2000 \
    >${PRINT_LOG} 2>&1
