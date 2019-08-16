#!/bin/bash

epoch_size=$1
lgs=$2
max_epoch=$3
lgsnohyp=${lgs/-/}

prefix=$HOME/twork/XLM_ctc

# tohle je debilni protoze XLM ma dim 4 * emb dim a my to mame 8x

export NGPU=`echo $CUDA_VISIBLE_DEVICES | sed 's/,/\n/g' | wc -l`


python -m torch.distributed.launch --nproc_per_node=$NGPU ../train.py \
       --exp_name nmt_${lgs}_with_bt \
       --dump_path $prefix/out/${lgs}_with_bt \
       --data_path $prefix/backtranslations/$lgsnohyp \
       --encoder_only false \
       --emb_dim 512 \
       --n_layers 6 \
       --n_heads 16 \
       --dropout 0.1 \
       --attention_dropout 0.1 \
       --gelu_activation true \
       --tokens_per_batch 2000 \
       --bptt 256 \
       --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.997,lr=0.0003 \
       --epoch_size $epoch_size \
       --max_epoch $max_epoch \
       --save_periodic 10 \
       --save_latest_ckpts 10 \
       --label_smoothing 0.1 \
       --eval_bleu true \
       --mt_steps "$lgs" \
       --lgs "$lgs" \
       --validation_metrics valid_${lgs}_mt_bleu \
       --stopping_criterion valid_${lgs}_mt_bleu,100
