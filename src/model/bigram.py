# src/models/bigram.py

import torch
import torch.nn as nn

from .base_model import BaseLanguageModel


class BigramModel(BaseLanguageModel):
    """
    一个简单的 Bigram 语言模型。
    预测下一个 token 只依赖于当前的 token。
    这本质上是一个查找表。
    """

    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        # 模型的全部就是一个嵌入层。
        # 对于每个输入的 token index，它会直接输出一个大小为 vocab_size 的向量，
        # 这个向量就是预测下一个 token 的 logits。
        self.embedding_table = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 输入张量, 形状为 (B, T) or (batch_size, context_window)

        Returns:
            torch.Tensor: Logits 张量, 形状为 (B, T, V) or (batch_size, context_window, vocab_size)
        """
        logits = self.embedding_table(x)
        return logits
