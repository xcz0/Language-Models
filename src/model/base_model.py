# src/models/base_model.py

import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLanguageModel(nn.Module, ABC):
    """
    所有语言模型的抽象基类。
    它继承自 nn.Module 和 ABC (Abstract Base Class)。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @abstractmethod
    def forward(self, x):
        """
        定义模型的前向传播。
        子类必须实现此方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, context_window)

        Returns:
            torch.Tensor: 模型的输出 logits，形状为 (batch_size, context_window, vocab_size)
        """
        pass
