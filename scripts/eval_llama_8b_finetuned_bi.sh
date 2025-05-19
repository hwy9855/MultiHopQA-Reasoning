#!/bin/bash

DATA_PATH="data/musique_ans_v1.0_dev.jsonl"
FINETUNED_MODEL_PATH="finetuned_models/llama_8b_bi"
HF_TOKEN="hf_..."

modes=("ft_llama")

for mode in "${modes[@]}"
do
    # original order
    python inference.py \
        --model_name $FINETUNED_MODEL_PATH \
        --model_type "llama" \
        --data_path $DATA_PATH \
        --mode $mode \
        --bi_attn \
        --token $HF_TOKEN

    doc_orders=("forward" "backward")

    # context permutation
    for doc_order in "${doc_orders[@]}"
    do
        python inference.py \
            --model_name $FINETUNED_MODEL_PATH \
            --model_type "llama" \
            --data_path $DATA_PATH \
            --mode $mode \
            --doc_order $doc_order \
            --bi_attn \
            --token $HF_TOKEN
    done
done