export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
export CUDA_VISIBLE_DEVICES='0,1'

ckpt_path=

# ft
python src/entry_point/interface.py \
    --ckpt_path $ckpt_path \
    --llama \
    --local_rank 0 \
    # --use_lora \
    # --lora_path
