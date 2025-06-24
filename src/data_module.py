# src/data_module.py

import json
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Optional

from src.tokenizer import CharTokenizer


class PoetryDataModule(LightningDataModule):
    def __init__(
        self, data_path: str, tokenizer_path: str, batch_size: int, context_window: int
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.context_window = context_window

        # 保存超参数，这使得在checkpoint中可以访问它们
        self.save_hyperparameters()

        self.tokenizer = CharTokenizer()
        self.full_text = ""
        self.encoded_data = None

    def prepare_data(self):
        """
        在单个进程上执行的操作：下载、分词等。
        这里我们用来构建和保存分词器词汇表。
        """
        # 1. 从JSON加载所有诗歌内容
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. 将所有诗歌的 'contents' 字段拼接成一个大字符串
        self.full_text = "\n".join([item["contents"] for item in data])

        # 3. 基于全量文本构建分词器并保存
        print("Fitting tokenizer...")
        self.tokenizer.fit(self.full_text)
        self.tokenizer.save(self.tokenizer_path)

    def setup(self, stage: Optional[str] = None):
        """
        在所有进程上执行的操作：加载分词器、创建数据集、执行拆分。
        """
        # 1. 加载分词器和全量文本（如果在不同进程中 prepare_data 未运行）
        self.tokenizer.load(self.tokenizer_path)
        if not self.full_text:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.full_text = "\n".join([item["contents"] for item in data])

        # 2. 将整个文本编码成一个大的整数列表
        self.encoded_data = torch.tensor(
            self.tokenizer.encode(self.full_text), dtype=torch.long
        )

        # 3. 创建输入(x)和目标(y)
        # 我们的目标是预测下一个字符，所以 y 是 x 向左移动一个位置
        # 例如，如果 context_window=4, x=[0,1,2,3], y=[1,2,3,4]
        num_sequences = len(self.encoded_data) - self.context_window

        # 我们需要创建一个包含 (x, y) 对的数据集
        # 这里为了简化，我们直接在 DataLoader 中使用一个简单的 TensorDataset
        # 注意：这里我们还没有进行训练/验证集划分，将在后续实现
        # 为了演示，我们先使用全量数据
        X = torch.stack(
            [
                self.encoded_data[i : i + self.context_window]
                for i in range(num_sequences)
            ]
        )
        Y = torch.stack(
            [
                self.encoded_data[i + 1 : i + self.context_window + 1]
                for i in range(num_sequences)
            ]
        )

        # 4. 创建数据集并进行拆分
        dataset = TensorDataset(X, Y)

        # 90% 训练, 10% 验证
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        print(f"Total sequences: {len(dataset)}")
        print(f"Training sequences: {len(self.train_dataset)}")
        print(f"Validation sequences: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )
