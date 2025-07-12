# src/data/datamodule.py

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from typing import List, Optional
from pathlib import Path

# 从同级目录的 dataset.py 文件中导入 CharDataset
from .dataset import CharDataset


class CharDataModule(pl.LightningDataModule):
    """
    一个 PyTorch Lightning DataModule，用于封装所有与字符数据相关的操作。
    """

    def __init__(
        self,
        data_dir: str,
        input_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        test_set_size: int = 1000,
    ):
        """
        初始化 DataModule。

        Args:
            input_file (str): 包含每行一个单词的输入文本文件路径。
            batch_size (int): 每个批次的样本数量。
            num_workers (int): 用于数据加载的子进程数量。
            test_set_size (int): 用于验证集的样本数量。
        """
        super().__init__()
        self.data_dir = data_dir
        self.input_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_set_size = test_set_size

        # 这些属性将在 setup() 方法中被赋值
        self.words: Optional[List[str]] = None
        self.chars: Optional[List[str]] = None
        self.max_word_length: Optional[int] = None
        self.train_dataset: Optional[Subset[CharDataset]] = None
        self.test_dataset: Optional[Subset[CharDataset]] = None

    @property
    def vocab_size(self) -> int:
        """方便地获取词汇表大小。"""
        if self.chars is None:
            raise RuntimeError(
                "DataModule has not been set up yet. Call setup() first."
            )
        return len(self.chars) + 1

    @property
    def block_size(self) -> int:
        """方便地获取模型的上下文/块大小。"""
        if self.max_word_length is None:
            raise RuntimeError(
                "DataModule has not been set up yet. Call setup() first."
            )
        return self.max_word_length + 1

    def prepare_data(self):
        """
        在单个进程上执行的操作，例如下载数据。
        由于我们使用本地文件，此方法中无需执行任何操作。
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        在每个GPU进程上执行的操作，包括数据预处理和数据集拆分。

        Args:
            stage (Optional[str]): 'fit' 或 'test'。
        """
        # 防止重复设置
        if self.train_dataset and self.test_dataset:
            return

        # 1. 加载和预处理数据
        file = Path(self.data_dir) / self.input_file
        with open(file, "r") as f:
            data = f.read()
        words = data.splitlines()
        words = [w.strip() for w in words]
        words = [w for w in words if w]
        self.words = words

        # 2. 计算数据集元数据
        self.chars = sorted(list(set("".join(self.words))))
        self.max_word_length = max(len(w) for w in self.words)

        print("--- Dataset Stats ---")
        print(f"Number of examples in the dataset: {len(self.words)}")
        print(f"Max word length: {self.max_word_length}")
        print(f"Number of unique characters: {len(self.chars)}")
        print(f"Vocabulary: {''.join(self.chars)}")
        print("---------------------")

        # 3. 将 self.words 转换为 CharDataset 类型
        full_dataset = CharDataset(self.words, self.chars, self.max_word_length)

        # 4. 拆分数据集为训练集和验证集
        test_size = min(self.test_set_size, int(len(self.words) * 0.1))
        train_size = len(self.words) - test_size

        # 使用固定的生成器以保证拆分的可复现性
        self.train_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(
            f"Split dataset into {len(self.train_dataset)} training and {len(self.test_dataset)} test examples."
        )

    def train_dataloader(self) -> DataLoader:
        """创建并返回训练数据加载器。"""
        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule has not been set up yet. Call setup() first."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """创建并返回验证数据加载器。"""
        if self.test_dataset is None:
            raise RuntimeError(
                "DataModule has not been set up yet. Call setup() first."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
