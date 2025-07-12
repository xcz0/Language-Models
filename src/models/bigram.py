# src/models/bigram.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

# 从同级目录的 base.py 中导入基类和配置类
from .base import LitBaseModel, ModelConfig


class Bigram(LitBaseModel):
    """
    Bigram (二元) 语言模型。

    这是最简单的神经语言模型，它仅通过一个可训练的查找表
    来预测基于前一个字符的下一个字符的 logits。
    它的上下文窗口 `block_size` 实际上是 1，但它能够处理
    由 DataModule 提供的更长的序列输入，只是在预测时只使用最后一个字符。
    """

    def __init__(self, config: ModelConfig, **kwargs):
        """
        初始化 Bigram 模型。

        Args:
            config (ModelConfig): 模型配置对象。
            **kwargs: 其他传递给基类的参数，如 learning_rate。
        """
        super().__init__(config, **kwargs)

        # Bigram 模型的核心就是一个大小为 (vocab_size, vocab_size) 的查找表。
        self.logits_table = nn.Parameter(
            torch.zeros((config.vocab_size, config.vocab_size))
        )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Bigram 模型的前向传播。

        Args:
            idx (torch.Tensor): 输入的 token 索引，形状为 (B, T)。
            targets (torch.Tensor, optional): 目标的 token 索引，形状为 (B, T)。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: logits 和 loss。
        """
        # 直接通过输入索引从查找表中获取 logits。
        # idx (B, T) -> logits (B, T, vocab_size)
        logits = self.logits_table[idx]

        loss = None
        if targets is not None:
            # 使用基类中定义的辅助函数计算损失
            loss = self._calculate_loss(logits, targets)

        return logits, loss

    def get_block_size(self) -> int:
        """
        覆盖基类方法。Bigram 模型的有效上下文大小为 1。
        尽管我们仍然可以从配置中返回 block_size 以保持与数据管道的兼容性，
        但明确指出其理论大小是有益的。
        """
        return 1
