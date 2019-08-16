epoch_size=${1:-50000}
stopping_crit=${2:-100}
lgs=${3:-de-en}
max_epoch=${4:-100}

lgsnohyp=${lgs/-/}

export NGPU=8; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU ../train.py \
  --exp_name nmt_${lgsnohyp}_trafobase \
  --dump_path ${PT_OUTPUT_DIR} \
  --data_path ${PT_DATA_DIR} \
  --lgs "$lgs" \
  --encoder_only false \
  --emb_dim 512 \
  --n_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --attention_dropout 0.1 \
  --tokens_per_batch 1000 \
  --bptt 256 \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --epoch_size $epoch_size \
  --max_epoch $max_epoch \
  --save_periodic 5 \
  --label_smoothing 0.1 \
  --eval_bleu true \
  --mt_steps "$lgs" \
  --validation_metrics valid_${lgs}_mt_bleu \
  --stopping_criterion valid_${lgs}_mt_bleu,$stopping_crit
