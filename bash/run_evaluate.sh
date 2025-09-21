#!/bin/bash
# 设置环境变量
#	--use_assistant_model aqua strategyqa gsm8k math
#	--att_file_name ..\/TransformerBlock.bin \
#	--print_response \ --num_return_sequences 10 \
#/home/LLM/Qwen2.5-7B-Instruct /home/LLM/Qwen3-8B
#/home/LLM/Qwen2.5-1.5B-Instruct /home/LLM/Qwen3-0.6B
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2,3
python -u evaluate.py \
	--base_model_id /home/LLM/Qwen3-8B \
	--assistant_model_id /home/LLM/Qwen3-0.6B \
	--params_file_name ../projection.bin \
	--att_file_name ../TransformerBlock.bin \
	--batch_size 4 \
	--task_name gsm8k \
	--num_thought_tokens 2 \
	--num_return_sequences 10 \
	--print_response \