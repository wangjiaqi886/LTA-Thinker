#!/bin/bash
# 设置环境变量
#	--use_assistant_model aqua strategyqa gsm8k
#	--att_file_name ..\TransformerBlock.bin \
#	--print_response \
#/home/LLM/Qwen2.5-7B-Instruct /home/LLM/Qwen3-8B
#/home/LLM/Qwen2.5-1.5B-Instruct /home/LLM/Qwen3-0.6B


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    train.py \
    --large_model_id /home/LLM/Qwen3-8B \
    --small_model_id /home/LLM/Qwen3-0.6B \
    --output_name qwen3-gsm8k-onlyliner-differthoughttokens-assistant-nohidden-lr8e5-con-kl_loss\
    --gradient_accumulation_steps 4 \
    --batch_size 4 \
    --task_name gsm8k \
    --learning_rate 1e-4 \
    --num_thought_tokens 32 \
    --n_epochs 10.0 \