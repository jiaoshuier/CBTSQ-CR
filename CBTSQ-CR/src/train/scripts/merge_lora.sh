#! /bin/bash


CUDA_VISIBLE_DEVICES=0 CBTSQ-CR/src/train/src/merge_llama_with_lora.py \
    --model_name_or_path CBTSQ-CR/model_path/Llama-2-7b-chat-hf \
    --output_path CBTSQ-CR/Merged_model/SoCBT \
    --lora_path ...l/BELLE/saved_models/llama2_raw_001/checkpoint-528 \
    --llama