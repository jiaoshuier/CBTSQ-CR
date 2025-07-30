#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1'
export WANDB_PROJECT='llama2_raw'
export WANDB_RUN_ID='001'
export WANDB_RESUME=allow
#export ABS_PATH='abs'
export ABS_PATH=''
export PYTHONPATH="$ABS_PATH/BELLE/train"
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
model_name_or_path='CBTSQ-CR/model_path/Llama-2-7b-chat-hf' # or bloomz-7b1-mt

train_file=CBTSQ-CR/src/train/scripts/data/train.json
validation_file=CBTSQ-CR/src/train/scripts/data/val.json
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024


# LoRA without 8bit
torchrun --nproc_per_node 2 CBTSQ-CR/src/train/scripts/src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --load_best_model_at_end True \
    --deepspeed CBTSQ-CR/configs/deepspeed_config_stage3.json \
    --lora_config CBTSQ-CR/configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 3 \
    --num_train_epochs 3 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --report_to tensorboard
    # --use_flash_attention
    # --resume_from_checkpoint ...
