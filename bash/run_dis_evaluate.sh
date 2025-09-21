#!/bin/bash
# 延时3.5小时后执行
#echo "脚本将在3.5小时后开始执行..."
#echo "当前时间: $(date)"
#echo "预计开始时间: $(date -d '+3 hours 30 minutes')"
#sleep 12600  # 3.5小时 = 3.5 * 3600 = 12600秒

#echo "延时结束，开始执行评估任务..."
#echo "实际开始时间: $(date)"
# ==============================================================================
# 1. 准备阶段: 设置变量并分配 Checkpoints 到子目录
# ==============================================================================

# 基础目录
result_dir=""

# --- 在这里设置你想使用的GPU列表! ---
gpus_to_use=(0 1 2 3 4 5 6 7)

# 程序将根据数组的长度自动确定并行任务的数量和子目录数量
num_gpus=${#gpus_to_use[@]}

# 检查基础目录是否存在
if [ ! -d "$result_dir" ]; then
    echo "错误: 结果目录未找到: $result_dir"
    exit 1
fi

echo "阶段 1: 准备并分配 Checkpoints..."
echo "将使用 $num_gpus 个 GPU: ${gpus_to_use[*]}"
echo "--------------------------------------------------"

mapfile -t source_checkpoints < <(find "$result_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

if [ ${#source_checkpoints[@]} -gt 0 ]; then
    echo "在 '$result_dir' 中找到 ${#source_checkpoints[@]} 个 checkpoint 目录，现在开始分配..."
    for i in $(seq 0 $((num_gpus - 1))); do
        mkdir -p "$result_dir/part_$i"
    done
    i=0
    for ckpt_path in "${source_checkpoints[@]}"; do
        target_part_dir="$result_dir/part_$((i % num_gpus))"
        echo "移动 '$(basename "$ckpt_path")' -> 'part_$((i % num_gpus))'"
        mv -n "$ckpt_path" "$target_part_dir/"
        i=$((i + 1))
    done
    echo "Checkpoint 分配完成。"
else
    echo "主目录中未发现 'checkpoint-*' 文件，假设它们已被分配。"
    echo "将直接在 'part_*' 子目录中进行评估。"
fi
echo


# ==============================================================================
# 2. 执行阶段: 在多个 GPU 上并行评估 (带日志记录)
# ==============================================================================

# (run_evaluation 函数保持不变)
run_evaluation() {
    local target_dir=$1
    local gpu_id=$2
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "[GPU $gpu_id] 开始处理目录: $target_dir"
    if [ ! -d "$target_dir" ]; then
        echo "[GPU $gpu_id] 错误: 目录不存在: $target_dir"
        return
    fi
    mapfile -t checkpoint_dirs < <(ls -v -d "$target_dir"/checkpoint-*/ 2>/dev/null)
    if [ ${#checkpoint_dirs[@]} -eq 0 ]; then
        echo "[GPU $gpu_id] 在 '$target_dir' 中没有找到任何 checkpoint 目录，任务完成。"
        return
    fi
    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
        if [ -d "$checkpoint_dir" ]; then
            local params_file="${checkpoint_dir}projection.bin"
#            local att_file="${checkpoint_dir}TransformerBlock.bin"
#		    if [ ! -f "$params_file" ] || [ ! -f "$att_file" ]; then
#			    echo "[GPU $gpu_id] 在 $(basename "$checkpoint_dir") 中缺少必要文件，跳过..."
#			    continue
#		    fi
            echo "--------------------------------------------------------------------------------"
            echo "[GPU $gpu_id] 正在评估 checkpoint: $(basename "$checkpoint_dir")"
            echo "--------------------------------------------------------------------------------"
            python evaluate.py \
                --base_model_id /home/LLM/Qwen3-8B \
                --assistant_model_id /home/LLM/Qwen3-0.6B \
                --params_file_name "$params_file" \
                --batch_size 8 \
                --task_name gsm8k \
                --num_thought_tokens 4 \
                --use_assistant_model
                
            echo "[GPU $gpu_id] 完成评估: $(basename "$checkpoint_dir")"
            echo
        fi
    done
    echo "[GPU $gpu_id] 已处理完目录 '$target_dir' 中的所有 checkpoints。"
}

echo "阶段 2: 启动并行评估，日志将输出到文件中..."
echo "================================================================================"

# 循环启动后台评估进程，并将每个进程的输出重定向到单独的日志文件
for i in $(seq 0 $((num_gpus - 1))); do
    part_dir="$result_dir/part_$i"
    gpu_id=${gpus_to_use[$i]}

    # --- 新增：定义日志文件名 ---
    log_file="$result_dir/evaluation_part_${i}_gpu_${gpu_id}.log"

    # --- 修改：在后台符&之前，添加 &> "$log_file" 进行重定向 ---
    echo "分发任务: '$part_dir' -> [GPU $gpu_id]。日志将保存在: $log_file"
    run_evaluation "$part_dir" "$gpu_id" &> "$log_file" &
done

echo
echo "已为 $num_gpus 个 GPU (${gpus_to_use[*]}) 启动了所有评估任务。"
echo "您现在可以使用 'tail -f' 命令来实时监控日志文件。"
echo "等待所有任务完成... (这可能需要很长时间)"

wait

echo "================================================================================"
echo "所有评估任务均已完成。"
echo "================================================================================"