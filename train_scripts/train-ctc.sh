epoch_size=${1:-50000}
stopping_crit=${2:-100}
lgs=${3:-cs-en}
max_epoch=${4:-100}

lgsnohyp=${lgs/-/}

#export NGPU=6; CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU ../train.py  \

export NGPU=1; CUDA_VISIBLE_DEVICES=0 python ../train.py  \
  --exp_name test_ctc \
  --dump_path dumpCTC \
  --data_path /home/t-jihelc/work/cuni/test_inp \
  --lgs "$lgs" \
  --encoder_only true \
  --emb_dim 512 \
  --n_layers 12 \
  --n_heads 8 \
  --dropout 0.1 \
  --attention_dropout 0.1 \
  --batch_size 16 \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --epoch_size $epoch_size \
  --max_epoch $max_epoch \
  --save_periodic 5 \
  --label_smoothing 0.0 \
  --eval_bleu true \
  --ctc_steps "$lgs" \
  --validation_metrics valid_${lgs}_mt_bleu \
  --stopping_criterion valid_${lgs}_mt_bleu,$stopping_crit \
  --ctc_model true \
  --ctc_split_factor 3 \
  --ctc_split_after_layer 5 \
  --ctc_use_inner_attention true \
  --ctc_add_pos_emb_after_split true
