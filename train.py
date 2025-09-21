import argparse
import torch
import os
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from fastNLP import logger  # 直接使用fastNLP的logger

from data_loader import (
    GSM8KLoader,
    StrategyQALoader,
    AugASDivLoader,
    AQuALoader,
    DULoader,
    MATHLoader,
)
from llm_model import EfficientSoftCoTFromSmallModel, MappingLayerTrainer
from utils import (
    pre_process_strategy_qa,
    pre_process_gsm8k,
    pre_process_aqua,
    pre_process_du,
    pre_process_math,
    CustomDataCollator,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large_model_id", type=str, default="/home/LLM/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--small_model_id", type=str, default="/home/LLM/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument("--params_file_name", type=str, default=None)
    parser.add_argument("--att_file_name", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_name", type=str, default="test_rebulid_4gpu_qwen")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--task_name",
        type=str,
        choices=[
            "gsm8k",
            "strategyqa",
            "asdiv-aug",
            "aqua",
            "du",
            "math",
        ],
        default="gsm8k",
    )
    parser.add_argument(
        "--use_assistant_model",
        action="store_true",
        default=False,
        help="Whether to use assistant model",
    )
    parser.add_argument("--num_thought_tokens", type=int, default=4)
    parser.add_argument("--n_epochs", type=float, default=10.0)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--tune_base_model", action="store_true", default=False)
    parser.add_argument("--tune_assistant_model", action="store_true", default=False)
    parser.add_argument("--max_len", type=int, default=-1)
    return parser.parse_args()


def setup_model_and_tokenizers(args):
    # 初始化分词器和特殊标记
    def get_special_tokens(model_id, type):
        if type == "base":
            if "Llama" in model_id:
                return {
                    "special_tokens": [
                        "<|end_of_text|>",
                        "<|reserved_special_token_0|>",
                        "<|reserved_special_token_1|>",
                    ],
                    "backbone": "llama",
                }
            elif "Qwen2.5" in model_id:
                return {
                    "special_tokens": ["<|endoftext|>", "<|box_start|>", "<|box_end|>"],
                    "backbone": "qwen",
                }
            elif "Qwen3" in model_id:
                return {
                    "special_tokens": ["<|endoftext|>", "<think>", "</think>"],
                    "backbone": "qwen3",
                }
            raise NotImplementedError(f"Unsupported model: {model_id}")
        elif type == "assistant":
            if "Llama" in model_id:
                return {
                    "special_tokens": [
                        "<|end_of_text|>",
                        "<|reserved_special_token_0|>",
                        "<|reserved_special_token_1|>",
                    ],
                    "backbone": "llama",
                }
            elif "Qwen2.5" in model_id:
                return {
                    "special_tokens": ["<|endoftext|>", "<|box_start|>", "<|box_end|>"],
                    "backbone": "qwen",
                }
            elif "Qwen3" in model_id:
                return {
                    "special_tokens": ["<|endoftext|>", "<think>", "</think>"],
                    "backbone": "qwen3",
                }
            raise NotImplementedError(f"Unsupported model: {model_id}")
        return None

    # base_tokenizer = AutoTokenizer.from_pretrained(args.large_model_id)
    # assistant_tokenizer = AutoTokenizer.from_pretrained(args.small_model_id)

    base_config = get_special_tokens(args.large_model_id, "base")
    assistant_config = get_special_tokens(args.small_model_id, "assistant")
    model_type = (args.use_assistant_model, base_config["backbone"])

    model = EfficientSoftCoTFromSmallModel(
        args.small_model_id,
        args.large_model_id,
        args.num_thought_tokens,
        path_to_projection_module=args.params_file_name,
        path_to_attention_module=args.att_file_name,
        tune_base_model=args.tune_base_model,
        tune_assistant_model=args.tune_assistant_model,
        run_type="train",
        model_type=model_type,
    )
    base_tokenizer = model.base_tokenizer
    assistant_tokenizer = model.assistant_tokenizer

    # 打印参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable/Total parameters: {trainable}/{total}")
    return model, base_tokenizer, assistant_tokenizer, base_config, assistant_config


def load_and_preprocess_data(
    args, base_tokenizer, assistant_tokenizer, base_config, assistant_config
):
    # 数据加载器映射
    loader_map = {
        "gsm8k": (GSM8KLoader, pre_process_gsm8k),
        "strategyqa": (StrategyQALoader, pre_process_strategy_qa),
        "asdiv-aug": (AugASDivLoader, pre_process_gsm8k),
        "aqua": (AQuALoader, pre_process_aqua),
        "du": (DULoader, pre_process_du),
        "math": (MATHLoader, pre_process_math),
    }
    loader_cls, preprocess_fn = loader_map[args.task_name]

    # 加载数据集
    db = loader_cls().load()
    train_ds = db["train"]
    eval_ds = db["dev"]
    # train_ds = db.get_dataset('train')
    # eval_ds = db.get_dataset('dev')

    if args.k_shot > 0:
        train_ds = train_ds[: args.k_shot]

    # 预处理参数
    preprocess_params = {
        "tokenizer": base_tokenizer,
        "assistant_tokenizer": assistant_tokenizer,
        "num_thought_tokens": args.num_thought_tokens,
        "add_bot_eot": True,
        "base_special_token": base_config["special_tokens"],
        "assistant_special_token": assistant_config["special_tokens"],
        "base_backbone": base_config["backbone"],
        "assistant_backbone": assistant_config["backbone"],
        "max_len": args.max_len,
    }

    # 数据预处理
    def process_dataset(dataset, split):
        return Dataset.from_pandas(
            pd.DataFrame(
                [
                    preprocess_fn(ins, **preprocess_params, split=split)
                    for ins in tqdm(dataset, desc=f"Preprocess {split} set")
                ]
            )
        )

    def process_dataset_optimized(dataset, split):
        # 将数据转换为Dataset对象
        if not isinstance(dataset, Dataset):
            dataset = Dataset.from_pandas(pd.DataFrame(dataset))

        # 定义预处理函数
        def preprocess_function(examples):
            processed = []
            for i in range(
                len(
                    examples["question"]
                    if "question" in examples
                    else list(examples.keys())[0]
                )
            ):
                # 构建单个样本
                sample = {k: v[i] for k, v in examples.items()}
                processed_sample = preprocess_fn(
                    sample, **preprocess_params, split=split
                )
                processed.append(processed_sample)

            # 重新组织数据格式
            result = {}
            if processed:
                for key in processed[0].keys():
                    result[key] = [item[key] for item in processed]
            return result

        # 使用 map 方法进行并行处理
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=32,  # 64
            num_proc=min(mp.cpu_count(), 4),
            desc=f"Preprocess {split} set",
            remove_columns=dataset.column_names,
        )

        return processed_dataset

    return process_dataset_optimized(train_ds, "train"), process_dataset_optimized(
        eval_ds, "dev"
    )


def setup_directories(args):
    large_name = args.large_model_id.split("/")[-1]
    small_name = args.small_model_id.split("/")[-1]
    post_fix = f"{args.task_name}-{args.n_epochs}-{args.num_thought_tokens}-{large_name}-{small_name}"

    return {
        "output_dir": f"./results/{args.output_name}-{post_fix}",
        "log_dir": f"./logs/{args.output_name}-{post_fix}",
        "model_dir": f"./ckpt/{args.output_name}-{post_fix}",
    }


def main():
    args = parse_arguments()
    logger.info(f"Arguments: {args.__dict__}")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    # 初始化模型和分词器
    model, base_tokenizer, assistant_tokenizer, base_config, assistant_config = (
        setup_model_and_tokenizers(args)
    )
    model = model.to(local_rank)

    # 加载并预处理数据
    train_data, eval_data = load_and_preprocess_data(
        args, base_tokenizer, assistant_tokenizer, base_config, assistant_config
    )

    # 设置目录
    dirs = setup_directories(args)
    logger.info(f"Directories: {dirs}")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=dirs["output_dir"],
        overwrite_output_dir=True,
        eval_strategy="epoch",
        eval_steps=500,
        save_strategy="epoch",
        learning_rate=args.learning_rate,  ##default 2e-5 * num gpu
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.n_epochs,
        bf16=True,
        logging_dir=dirs["log_dir"],
        logging_steps=500,
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_num_workers=32,
        local_rank=-1,  # 分布式训练必须设置
        ddp_find_unused_parameters=True,
        save_only_model=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
    )

    # 初始化Trainer
    trainer = MappingLayerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=CustomDataCollator(base_tokenizer, assistant_tokenizer),
    )

    # 开始训练
    trainer.train()
    if trainer.args.process_index == 0:
        model.save_pretrained(dirs["model_dir"])
        logger.info(f'Finish training, save model to dir `{dirs["model_dir"]}`')

    # if torch.distributed.is_initialized():
    #     torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
