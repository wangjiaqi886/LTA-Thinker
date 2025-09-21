import re
import argparse

from tqdm import tqdm
import torch
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode
from pylatexenc.latex2text import LatexNodes2Text
from typing import Dict, Any, Tuple, List, Optional
from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger
from torch.utils.data import DataLoader

from llm_model import EfficientSoftCoTFromSmallModel
from data_loader import (
    GSM8KLoader,
    StrategyQALoader,
    AugASDivLoader,
    AQuALoader,
    DULoader,
    MATHLoader,
)
from utils import (
    pre_process_gsm8k,
    pre_process_strategy_qa,
    pre_process_aqua,
    pre_process_du,
    pre_process_math,
)
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class Config:
    """封装模型相关配置"""

    def __init__(self, args: argparse.Namespace):
        self.base_model_id = args.base_model_id
        self.assistant_model_id = args.assistant_model_id
        self.params_file_name = args.params_file_name
        self.att_file_name = args.att_file_name
        self.base_model_ckpt = (
            args.base_model_ckpt if args.base_model_ckpt != "None" else None
        )
        self.assistant_model_ckpt = (
            args.assistant_model_ckpt if args.assistant_model_ckpt != "None" else None
        )
        self.num_thought_tokens = args.num_thought_tokens
        self.num_return_sequences = args.num_return_sequences
        self.tune_base_model = args.tune_base_model
        self.tune_assistant_model = args.tune_assistant_model
        self.max_new_tokens = args.max_new_tokens
        self.use_assistant_model = args.use_assistant_model
        self.task_name = args.task_name
        self.batch_size = args.batch_size
        self.print_input = args.print_input
        self.print_response = args.print_response
        self.test_k = args.test_k
        self.seed = args.seed


def parse_arguments() -> Tuple[Config]:
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument(
        "--base_model_id", type=str, default="/home/LLM/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--assistant_model_id", type=str, default="/home/LLM/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--params_file_name",
        type=str,
        default="../checkpoint-234/projection.bin",
    )
    parser.add_argument(
        "--att_file_name",
        type=str,
        default=None,
    )
    parser.add_argument("--base_model_ckpt", type=str, default=None)
    parser.add_argument("--assistant_model_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_thought_tokens", type=int, default=4)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--tune_base_model", action="store_true", default=False)
    parser.add_argument("--tune_assistant_model", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--use_assistant_model",
        action="store_true",
        default=False,
        help="Whether to use assistant model",
    )

    # 任务参数
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
    parser.add_argument("--print_input", action="store_true", default=False)
    parser.add_argument("--print_response", action="store_true", default=False)
    parser.add_argument("--test_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    return Config(args)


def truncate_after_terminator(tensor, terminator):
    """
    截取张量中第一个出现的终止符（如eos token）之前的所有token。

    参数:
        tensor (torch.Tensor): 一维token ID张量
        terminator (int): 终止符token ID，如151645

    返回:
        torch.Tensor: 截取后的有效token序列
    """
    # 找到第一个等于terminator的位置
    mask = tensor == terminator
    indices = torch.nonzero(mask)

    if indices.size(0) == 0:
        # 如果没有找到terminator，保留整个序列
        return tensor
    else:
        first_index = indices[0, 0].item()
        return tensor[:first_index]


def get_special_tokens(model_id: str, type: str) -> tuple[list[str], str] | None:
    """获取模型特殊token和骨干类型"""
    if type == "base":
        if "Llama" in model_id:
            return [
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
            ], "llama"
        elif "Qwen2.5" in model_id:
            return ["<|endoftext|>", "<|box_start|>", "<|box_end|>"], "qwen"
        elif "Qwen3" in model_id:
            return ["<|endoftext|>", "<think>", "</think>"], "qwen3"
        raise NotImplementedError(f"Model {model_id} not supported")
    elif type == "assistant":
        if "Llama" in model_id:
            return [
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
            ], "llama"
        elif "Qwen2.5" in model_id:
            return ["<|endoftext|>", "<|box_start|>", "<|box_end|>"], "qwen"
        elif "Qwen3" in model_id:
            return ["<|endoftext|>", "<think>", "</think>"], "qwen3"
        raise NotImplementedError(f"Model {model_id} not supported")
    return None


def load_dataset(task_name: str, test_k: int = 0) -> Tuple[Any, callable]:
    """加载数据集和预处理方法"""
    loader_map = {
        "gsm8k": (GSM8KLoader, pre_process_gsm8k),
        "strategyqa": (StrategyQALoader, pre_process_strategy_qa),
        "asdiv-aug": (AugASDivLoader, pre_process_gsm8k),
        "aqua": (AQuALoader, pre_process_aqua),
        "du": (DULoader, pre_process_du),
        "math": (MATHLoader, pre_process_math),
        # 'math': (MathLoader, pre_process_math),  # 假设有
    }

    if task_name not in loader_map:
        raise NotImplementedError(f"Task {task_name} not supported")

    loader_class, preprocess_method = loader_map[task_name]
    db = loader_class().load()
    ds = db["test"]

    if test_k > 0:
        ds = ds[:test_k]

    return ds, preprocess_method


def setup_generation_config(base_model_id: str) -> GenerationConfig:
    """配置生成参数"""
    generation_config = GenerationConfig.from_pretrained(base_model_id)
    _, backbone = get_special_tokens(base_model_id, "base")

    if backbone == "llama":
        generation_config.pad_token_id = 128009
    elif backbone == "qwen3" or backbone == "qwen":
        generation_config.pad_token_id = 151643

    if backbone == "qwen3":
        # generation_config.top_p = 0.95
        # generation_config.top_k = 20
        # generation_config.temperature = 0.6
        # generation_config.min_p = 0

        # generation_config.top_p = 0.99
        # generation_config.temperature = 0.99

        generation_config.top_p = 0.8
        generation_config.top_k = 20
        generation_config.temperature = 0.7
        generation_config.min_p = 0

    elif backbone == "qwen2.5":
        # qwen2.5
        generation_config.top_p = 0.8
        generation_config.temperature = 0.7
        generation_config.repetition_penalty = 1.05
    else:
        generation_config.top_p = 1.0
        generation_config.top_k = 10
        generation_config.temperature = 0.5
    return generation_config


def extract_ground_truth(task_name: str, ins: str) -> Tuple[str]:
    if task_name in ["gsm8k", "asdiv-aug", "aqua"]:
        answer = ins["answer"].split("\n")[-1]
        assert answer.startswith("####")
        answer = answer.replace(",", "")
        if task_name in ["gsm8k", "asdiv-aug"]:
            if "." in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ["strategyqa"]:
        answer = ins["answer"]
    elif task_name in ["du"]:
        answer = ins["answer"].split("\n")[-1]
    elif task_name == "math":
        answer = ins["answer"].split("\n")[-1]
        assert answer.startswith("####")
        answer = str(answer[5:])
        answer = clean_latex_answer(answer)
        answer = normalize_answer(answer)
    else:
        raise NotImplementedError
    return answer


def get_inputs_embeds_with_parallel_support(
    model: nn.Module, inputs: Dict[str, torch.Tensor], print_input: bool
) -> torch.Tensor:
    if isinstance(model, nn.DataParallel):
        base_model = model.module.base_model
        get_embeds_func = model.module.get_inputs_embeds_for_base_model
    else:
        base_model = model.base_model
        get_embeds_func = model.get_inputs_embeds_for_base_model

    # 获取基础嵌入
    inputs_embeds = base_model.get_input_embeddings()(inputs["input_ids"])

    # 调用核心处理函数
    inputs_embeds = get_embeds_func(
        inputs["assistant_input_ids"],
        inputs["assistant_attention_mask"],
        inputs["input_ids"],
        inputs_embeds,
        inputs["thought_index"],
        inputs["q_and_step_index_list"],
        inputs["ans_ids"],
        print_input,
    )

    return inputs_embeds


def pad_tensor_list(tensor_list: list[torch.Tensor], value: int) -> list[torch.Tensor]:
    # max_len = max(t.size(1) for t in tensor_list)
    # padded_tensors = []
    # for tensor in tensor_list:
    #     current_len = tensor.size(1)
    #     pad_size = max_len - current_len
    #     padded_tensor = F.pad(tensor, (0, pad_size), value=value)
    #     padded_tensors.append(padded_tensor)
    # return padded_tensors

    max_len = max(t.size(1) for t in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        current_len = tensor.size(1)
        if current_len < max_len:
            pad_size = max_len - current_len
            padded_tensor = F.pad(tensor, (0, pad_size), value=value)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    # 将所有张量堆叠成一个批次
    return torch.cat(padded_tensors, dim=0)


def restore_original_list(aggregated_dict):
    if not aggregated_dict:
        return []

    keys = list(aggregated_dict.keys())
    first_key = keys[0]
    expected_length = len(aggregated_dict[first_key])

    for key in keys[1:]:
        if len(aggregated_dict[key]) != expected_length:
            raise ValueError(f"键 '{key}' 的列表长度不一致，无法还原")
    return [
        {key: aggregated_dict[key][i] for key in keys} for i in range(expected_length)
    ]


def extract_key_values(dict_list, base_tokenizer, assistant_tokenizer):
    if not dict_list:
        return {}

    # 使用字典推导式提取每个键的所有值
    # new_dict_list = {key: [d[key] for d in dict_list] for key in dict_list[0].keys()}
    # new_dict_list["input_ids"] = pad_tensor_list(
    #     new_dict_list["input_ids"], base_tokenizer.pad_token_id
    # )
    # new_dict_list["attention_mask"] = pad_tensor_list(
    #     new_dict_list["attention_mask"], 0
    # )
    # new_dict_list["assistant_input_ids"] = pad_tensor_list(
    #     new_dict_list["assistant_input_ids"], assistant_tokenizer.pad_token_id
    # )
    # new_dict_list["assistant_attention_mask"] = pad_tensor_list(
    #     new_dict_list["assistant_attention_mask"], 0
    # )

    # original_data = restore_original_list(new_dict_list)
    # return original_data

    batch_data = {}
    for key in dict_list[0].keys():
        if key in [
            "input_ids",
            "attention_mask",
            "assistant_input_ids",
            "assistant_attention_mask",
        ]:
            tensor_list = [d[key] for d in dict_list]
            if key in ["input_ids", "assistant_input_ids"]:
                pad_token_id = (
                    base_tokenizer.pad_token_id
                    if "assistant" not in key
                    else assistant_tokenizer.pad_token_id
                )
                batch_data[key] = pad_tensor_list(tensor_list, pad_token_id)
            else:
                batch_data[key] = pad_tensor_list(tensor_list, 0)
        elif key == "thought_index":
            # 将 thought_index 从 list of tensors 转换为单个 tensor
            tensor_list = [d[key] for d in dict_list]
            batch_data[key] = torch.cat(tensor_list, dim=0)
        else:
            batch_data[key] = [d[key] for d in dict_list]

    return batch_data


def prepare_model_inputs(
    model,
    batch,
    task_name: str,
    base_tokenizer: AutoTokenizer,
    assistant_tokenizer: AutoTokenizer,
    num_thought_tokens: int,
    preprocess_method: callable,
    base_special_token: List[str],
    assistant_special_token: List[str],
    base_backbone: str,
    assistant_backbone: str,
    seed: int,
    print_input: bool = False,
) -> Tuple[Dict, torch.Tensor, torch.Tensor, List[str]]:
    # ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor, List[str]]:
    if isinstance(model, nn.DataParallel):
        model_device = model.module.device
    else:
        model_device = model.device

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    input_list = []
    answers = []
    inputs_attention_mask = []
    inputs_embeds_all = []

    batch_transformed = []
    for i, (q, a) in enumerate(zip(batch["question"], batch["answer"])):
        item = {"question": q, "answer": a}
        if task_name == "strategyqa" and "facts" in batch:
            if i < len(batch["facts"]):
                item["facts"] = batch["facts"][i]
        batch_transformed.append(item)

    for ins in batch_transformed:
        answer = extract_ground_truth(task_name, ins)
        logger.info(f"Ground Truth Answer: {answer}")

        inputs = preprocess_method(
            ins,
            base_tokenizer,
            assistant_tokenizer,
            num_thought_tokens,
            add_bot_eot=(num_thought_tokens > 0),
            split="test",
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
            device=model_device,
        )
        if print_input:
            logger.info(
                f'Raw Inputs for Base Model: {base_tokenizer.decode(inputs["input_ids"][0])}'
            )
            # logger.info(f'Raw Inputs for Assistant Model: {assistant_tokenizer.decode(inputs["assistant_input_ids"][0])}')

        input_list.append(inputs)
        answers.append(answer)

    # input_list = extract_key_values(input_list, base_tokenizer, assistant_tokenizer)
    # for inputs in input_list:
    #     (inputs_embeds, _, _) = get_inputs_embeds_with_parallel_support(
    #         model, inputs, print_input
    #     )

    #     inputs_embeds_all.append(inputs_embeds)
    #     inputs_attention_mask.append(inputs["attention_mask"])
    # inputs_embeds_all = torch.cat(inputs_embeds_all, dim=0)
    # inputs_attention_mask = torch.cat(inputs_attention_mask, dim=0)
    # return input_list, inputs_embeds_all, inputs_attention_mask, answers

    batch_inputs = extract_key_values(input_list, base_tokenizer, assistant_tokenizer)
    with torch.no_grad():
        inputs_embeds, _, _ = get_inputs_embeds_with_parallel_support(
            model, batch_inputs, print_input
        )
    return batch_inputs, inputs_embeds, batch_inputs["attention_mask"], answers


def normalize_answer(answer):
    """标准化答案格式"""
    # 移除多余空格
    answer = re.sub(r"\s+", "", answer)
    # 或者统一添加空格
    # answer = re.sub(r'([(),])', r' \1 ', answer)
    # answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def clean_latex_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\left", ""),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    assert "\n" not in final_answer
    assert "\r" not in final_answer
    assert "\f" not in final_answer
    if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

    if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
        final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]

    if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

    if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
        final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    try:
        # 解析LaTeX字符串
        lw = LatexWalker(final_answer)
        # 提取节点信息，并进行初步解析
        nodelist, _, _ = lw.get_latex_nodes()

        # 使用内置的转换器将节点重新转换为文本
        # 这通常会产生一个更“干净”的LaTeX字符串
        converter = LatexNodes2Text()
        normalized_text = converter.nodelist_to_text(nodelist)

        return normalized_text

    except Exception as e:
        return f"解析错误: {e}"


def process_model_response(
    task_name, raw_model_answer, print_response, response_true, base_tokenizer, response
):
    if task_name in ["gsm8k", "asdiv-aug"]:
        cleaned_model_answer = raw_model_answer.replace(",", "")
        cleaned_model_answer = cleaned_model_answer.replace("%", "")
        cleaned_model_answer = cleaned_model_answer.replace("$", "")
    else:
        cleaned_model_answer = raw_model_answer

    match = re.findall(r"\s*([\d,]+(?:\.\d+)?)\s*", cleaned_model_answer)

    if task_name == "math":
        # 提取 \boxed{} 中的内容
        pattern = r"\\boxed\{((?:[^{}]++|\{(?:[^{}]++|\{[^{}]*+\})*+\})*+)\}"
        boxed_matches = re.findall(pattern, cleaned_model_answer)

        if boxed_matches:
            # 取最后一个 boxed 答案
            model_answer = boxed_matches[-1].strip()
            # 移除可能的 LaTeX 格式
            model_answer = clean_latex_answer(model_answer)
            model_answer = normalize_answer(model_answer)
        else:
            # 如果没有找到 boxed，尝试其他模式
            # 查找数字答案
            match = re.findall(r"\s*([\d,]+(?:\.\d+)?)\s*", cleaned_model_answer)
            if match:
                model_answer = match[-1].replace(",", "")
            else:
                model_answer = None

        if model_answer is None and not print_response:
            logger.info(
                f"None Model Answer ({response_true}): {base_tokenizer.decode(response)}"
            )
    elif task_name in ["gsm8k", "asdiv-aug"]:
        try:
            if match:
                last_match = match[-1]
                cleaned_match = last_match.replace(",", "")
                cleaned_match = cleaned_match.replace("%", "")
                cleaned_match = cleaned_match.replace("$", "")
                if "." in cleaned_match:
                    model_answer = round(float(cleaned_match), 2)
                else:
                    model_answer = int(cleaned_match)
            else:
                model_answer = None
            if model_answer is None and not print_response:
                logger.info(
                    f"None Model Answer ({response_true}): {base_tokenizer.decode(response)}"
                )
        except Exception as e:
            model_answer = None
            logger.error(f"Error: {e}")
    elif task_name in ["strategyqa"]:
        last_yes = re.search(r"\bsey\b", raw_model_answer.lower()[::-1])
        if last_yes is not None:
            last_yes = last_yes.start()
        else:
            last_yes = len(raw_model_answer)
        last_no = re.search(r"\bon\b", raw_model_answer.lower()[::-1])
        if last_no is not None:
            last_no = last_no.start()
        else:
            last_no = len(raw_model_answer)
        if last_yes == last_no == len(raw_model_answer):
            model_answer = None
        else:
            model_answer = last_yes < last_no
    elif task_name in ["aqua", "du"]:
        m_answer = re.search(r"\b[a-f]\b", raw_model_answer.lower()[::-1])
        if m_answer is not None:
            model_answer = m_answer.group(0).upper()
        else:
            model_answer = None
    else:
        raise NotImplementedError
    return model_answer


def main():
    config = parse_arguments()
    logger.info(f"Model config: {config.__dict__}")

    base_model_id = config.base_model_id
    assistant_model_id = config.assistant_model_id
    params_file_name = config.params_file_name
    att_file_name = config.att_file_name
    base_model_ckpt = config.base_model_ckpt
    assistant_model_ckpt = config.assistant_model_ckpt
    num_thought_tokens = config.num_thought_tokens
    num_return_sequences = config.num_return_sequences
    use_assistant_model = config.use_assistant_model

    task_name = config.task_name
    print_input = config.print_input
    print_response = config.print_response
    test_k = config.test_k
    batch_size = config.batch_size
    seed = config.seed
    tune_base_model = config.tune_base_model
    tune_assistant_model = config.tune_assistant_model
    max_new_tokens = config.max_new_tokens

    large_model_name = base_model_id.split("/")[-1]
    small_model_name = assistant_model_id.split("/")[-1]

    if base_model_ckpt in ["None"]:
        base_model_ckpt = None
    if assistant_model_ckpt in ["None"]:
        assistant_model_ckpt = None

    model_dtype = torch.bfloat16
    param_dtype = str(model_dtype)

    # base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token='your-huggingface-token')
    # assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id, token='your-huggingface-token')

    base_special_token, base_backbone = get_special_tokens(config.base_model_id, "base")
    assistant_special_token, assistant_backbone = get_special_tokens(
        config.assistant_model_id, "assistant"
    )

    model_type = (use_assistant_model, base_backbone)

    model = EfficientSoftCoTFromSmallModel(
        assistant_model_id,
        base_model_id,
        num_thought_tokens,
        tune_base_model=tune_base_model,
        tune_assistant_model=tune_assistant_model,
        path_to_projection_module=params_file_name,
        path_to_attention_module=att_file_name,
        path_to_small_language_model=assistant_model_ckpt,
        run_type="eval",
        model_type=model_type,
    )
    base_tokenizer = model.base_tokenizer
    assistant_tokenizer = model.assistant_tokenizer
    logger.info(f"Successfully Init Model `{model.__class__.__name__}`")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.eval()
        if model.module.assistant_model is not None:
            model.module.assistant_model.eval()
        model.module.base_model.eval()
    else:
        model.eval()
        if model.assistant_model is not None:
            model.assistant_model.eval()
        model.base_model.eval()

    ds, preprocess_method = load_dataset(config.task_name, config.test_k)
    logger.info(f"Successfully Load Dataset")
    generation_config = setup_generation_config(config.base_model_id)

    correct_count = 0
    total_tokens = 0  # 改为记录总token数
    total_sequences = 0
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    terminators = [
        base_tokenizer.eos_token_id,
    ]
    original_batch_size = dataloader.batch_size
    if base_backbone in ["llama"]:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    if base_backbone in ["qwen3"]:
        max_new_tokens = 4096
        # if task_name in ["du", "strategyqa"]:
        #     max_new_tokens /= 4
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        # for idx, ins in enumerate(tqdm(ds)):

        inputs, inputs_embeds, inputs_attention_mask, answer = prepare_model_inputs(
            model=model,
            batch=batch,
            task_name=task_name,
            base_tokenizer=base_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            num_thought_tokens=num_thought_tokens,
            preprocess_method=preprocess_method,
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
            seed=seed,
            print_input=print_input,
        )

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                outputs = model.module.base_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs_attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    generation_config=generation_config,
                    num_return_sequences=num_return_sequences,
                    use_cache=True,
                )
            else:
                outputs = model.base_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs_attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    generation_config=generation_config,
                    num_return_sequences=num_return_sequences,
                    use_cache=True,
                )
        _, output_length = outputs.shape
        if num_return_sequences > 1:
            batch_size = int(outputs.size(0) / num_return_sequences)
        else:
            batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, num_return_sequences, output_length)
        for index, out_batch in enumerate(outputs):
            model_answer_list = []
            model_answer_count = {}
            idx = batch_idx * original_batch_size + index
            for i in range(out_batch.shape[0]):
                # response = outputs[i][inputs['input_ids'].shape[-1]:]
                response = out_batch[i]
                response = truncate_after_terminator(response, terminators[0])
                total_tokens += response.shape[0]
                total_sequences += 1
                raw_model_answer = base_tokenizer.decode(
                    response, skip_special_tokens=True
                )

                # response_true = f"{batch_idx * batch_size + 1 + i}/{len(ds)}-{i + 1}/{batch_size}-{batch_idx + 1}/{len(dataloader)}"
                response_true = f"({idx + 1}-{i + 1}/{len(ds)})"

                model_answer = process_model_response(
                    task_name,
                    raw_model_answer,
                    print_response,
                    response_true,
                    base_tokenizer,
                    response,
                )

                model_answer_list.append(model_answer)
                if model_answer in model_answer_count and model_answer is not None:
                    model_answer_count[model_answer] += 1
                else:
                    model_answer_count[model_answer] = 1

            max_model_count = 0
            final_model_answer = None

            for k, v in model_answer_count.items():
                if v > max_model_count:
                    final_model_answer = k
                    max_model_count = v

            logger.info(f"Ground Truth Answer: {answer[index]}")
            logger.info(f"Model Answer: {final_model_answer}")
            is_correct = final_model_answer == answer[index]
            logger.info(f"Is Correct: {is_correct}")
            if is_correct:
                correct_count += 1
            else:
                if print_response:
                    logger.info(
                        f"Answer  ({response_true}): {base_tokenizer.decode(response)}<|end-of-response|>"
                    )
            logger.info(f"Correct Count: {correct_count}/{idx + 1}")
            Correct_Acc = correct_count / (idx + 1) if idx + 1 > 0 else 0
            logger.info(f"Correct_ACC: {Correct_Acc:.4f} ({Correct_Acc * 100:.2f}%)")
            logger.info(f'{"-" * 20}')
        del inputs, inputs_embeds, inputs_attention_mask, outputs, batch
        torch.cuda.empty_cache()
    avg_token = total_tokens / total_sequences if total_sequences > 0 else 0
    logger.info(f"Average Token Length: {avg_token:.2f}")
    logger.info(f"Total Sequences: {total_sequences}")
    logger.info(f"Total Tokens: {total_tokens}")

    # 计算最终正确率
    total_questions = len(ds)
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    logger.info(f"Final Results:")
    logger.info(f"Correct Count: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
