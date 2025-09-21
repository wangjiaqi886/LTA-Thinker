import os
import json
from typing import Union, Dict
import ast

import pandas as pd
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from datasets import DatasetDict, load_dataset
import os
from typing import Union, Dict


class MATHLoader:
    def __init__(self, base_path: str = "data/MATH"):
        self.base_path = base_path
        self.default_files = {
            "train": "math_gsmstyle.jsonl",
            "dev": "math500_gsm.jsonl",
            "test": "math500_gsm.jsonl",
        }

    def load(self, paths: str | dict = None) -> DatasetDict:
        # 处理路径输入
        if isinstance(paths, str):
            paths = {k: os.path.join(paths, v) for k, v in self.default_files.items()}
        elif paths is None:
            paths = {
                k: os.path.join(self.base_path, v)
                for k, v in self.default_files.items()
            }

        # 加载数据集并重命名拆分
        ds = load_dataset("json", data_files=paths)
        return DatasetDict(
            {
                "train": ds["train"],
                "dev": ds["dev"],  # 默认映射 dev 到 validation
                "test": ds["test"],
            }
        )



class GSM8KLoader:
    def __init__(self, base_path: str = "data/gsm8k"):
        self.base_path = base_path
        self.default_files = {
            "train": "train_socratic.jsonl",
            "dev": "test_socratic.jsonl",
            "test": "test_socratic.jsonl",
        }

    def load(self, paths: str | dict = None) -> DatasetDict:
        # 处理路径输入
        if isinstance(paths, str):
            paths = {k: os.path.join(paths, v) for k, v in self.default_files.items()}
        elif paths is None:
            paths = {
                k: os.path.join(self.base_path, v)
                for k, v in self.default_files.items()
            }

        # 加载数据集并重命名拆分
        ds = load_dataset("json", data_files=paths)
        return DatasetDict(
            {
                "train": ds["train"],
                "dev": ds["dev"],  # 默认映射 dev 到 validation
                "test": ds["test"],
            }
        )


class AQuALoader:
    def __init__(self, base_path: str = "data/AQuA-master"):
        self.base_path = base_path
        self.default_files = {
            "train": "gsm_style_train.jsonl",
            "dev": "gsm_style_dev.jsonl",
            "test": "gsm_style_test.jsonl",
        }

    def load(self, paths: str | dict = None) -> DatasetDict:
        # 处理路径输入
        if isinstance(paths, str):
            paths = {k: os.path.join(paths, v) for k, v in self.default_files.items()}
        elif paths is None:
            paths = {
                k: os.path.join(self.base_path, v)
                for k, v in self.default_files.items()
            }

        # 加载数据集并重命名拆分
        ds = load_dataset("json", data_files=paths)
        return DatasetDict(
            {
                "train": ds["train"],
                "dev": ds["dev"],  # 默认映射 dev 到 validation
                "test": ds["test"],
            }
        )


class DULoader:
    def __init__(self, base_path: str = "data/du"):
        self.base_path = base_path
        self.default_files = {
            "train": "date_understanding_gsm_style.jsonl",
            "dev": "date_understanding_gsm_style.jsonl",
            "test": "date_understanding_gsm_style.jsonl",
        }

    def load(self, paths: str | dict = None) -> DatasetDict:
        # 处理路径输入
        if isinstance(paths, str):
            paths = {k: os.path.join(paths, v) for k, v in self.default_files.items()}
        elif paths is None:
            paths = {
                k: os.path.join(self.base_path, v)
                for k, v in self.default_files.items()
            }

        # 加载数据集并重命名拆分
        ds = load_dataset("json", data_files=paths)
        return DatasetDict(
            {
                "train": ds["train"],
                "dev": ds["dev"],  # 默认映射 dev 到 validation
                "test": ds["test"],
            }
        )


class StrategyQALoader:
    def __init__(self, base_path: str = "data/strategyqa"):
        self.base_path = base_path
        self.default_files = {
            "train": "strategyqa_train.jsonl",
            "dev": "strategyqa_dev.jsonl",
            "test": "strategyqa_dev.jsonl",
        }

    def load(self, paths: str | dict = None) -> DatasetDict:
        # 处理路径输入
        if isinstance(paths, str):
            paths = {k: os.path.join(paths, v) for k, v in self.default_files.items()}
        elif paths is None:
            paths = {
                k: os.path.join(self.base_path, v)
                for k, v in self.default_files.items()
            }

        # 加载数据集并重命名拆分
        ds = load_dataset("json", data_files=paths)
        return DatasetDict(
            {
                "train": ds["train"],
                "dev": ds["dev"],  # 默认映射 dev 到 validation
                "test": ds["test"],
            }
        )


class AugASDivLoader(GSM8KLoader):
    def load(
        self, paths: Union[str, Dict[str, str]] = "/path/to/data/dir"
    ) -> DataBundle:
        if isinstance(paths, str):
            paths = {
                "train": os.path.join(paths, "aug-train.jsonl"),
                "dev": os.path.join(paths, "aug-dev.jsonl"),
                "test": os.path.join(paths, "aug-dev.jsonl"),
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
