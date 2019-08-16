#!/bin/bash

src=$1
tgt=$2
start=$3
length=$4

end=$((start+length))

python translate_pth.py \
       --input ${PT_DATA_DIR}/train.${src}.pth \
       --model_path ${PT_DATA_DIR}/model.pth \
       --output_path ${PT_OUTPUT_DIR}/output.$start-$end.sp \
       --dump_path ${PT_OUTPUT_DIR} \
       --src_lang $src \
       --tgt_lang $tgt \
       --exp_name translation \
       --beam_size 4 \
       --length_penalty 0.6 \
       --batch_size 128 \
       --subset_start $start \
       --subset_end $end
