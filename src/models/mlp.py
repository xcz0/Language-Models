# src/models/mlp.py
"""
MLP (Multi-Layer Perceptron) 语言模型的 Lightning 实现。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import LitBaseModel, ModelConfig, AdamConfig


class MLP(LitBaseModel):
    """
    基于多层感知机的神经语言模型。

    参考 Bengio 等人 2003 年的经典论文。它通过将固定大小（`block_size`）上下文窗口内的词嵌入向量进行拼接，然后将这个拼接后的大向量输入到一个 MLP 中来预测下一个 token。
    """

    def __init__(
        self,
        config: ModelConfig,
        optim_config: Optional[AdamConfig] = None,
    ):
        """
        初始化 MLP 模型。

        Args:
            config (ModelConfig): 包含模型架构参数的配置对象。
            optim_config (AdamConfig, optional): 优化器配置。
        """
        super().__init__(config, optim_config)

        # 词嵌入表 (Token Embeddings Table)
        # 词汇表大小需要 +1，是为了给一个特殊的 <BLANK> token 留出位置。
        # 当上下文窗口延伸到序列开始之前时，会使用这个 token。
        self.wte = nn.Embedding(self.config.vocab_size + 1, self.config.n_embd)

        # 多层感知机 (MLP)
        self.mlp = nn.Sequential(
            # 输入层：将 block_size 个 n_embd 维度的向量拼接成一个大向量。
            nn.Linear(self.config.block_size * self.config.n_embd, self.config.n_embd2),
            # 激活函数
            nn.Tanh(),
            # 输出层：输出每个词的 logits
            nn.Linear(self.config.n_embd2, self.config.vocab_size),
        )

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MLP 模型的前向传播。

        Args:
            idx (torch.Tensor): 输入的 token 索引张量，形状为 (B, T)，其中 T 必须等于 block_size。
            targets (torch.Tensor, optional): 目标的 token 索引张量，形状为 (B, T)。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - logits: 模型的输出，形状为 (B, T, vocab_size)。
            - loss: 如果提供了 targets，则为计算出的交叉熵损失；否则为 None。
        """
        # 原始代码中，MLP 的 forward 逻辑比较特殊，它通过一个循环和 torch.roll
        # 来为每个时间步构建上下文。我们在这里忠实地复现这个逻辑。
        # 注意：这种实现方式在概念上是正确的，但可能不是最高效的，
        # 一个更常见的 MLP 语言模型实现会先将输入展平。

        embs = []
        # 通过循环和滚动操作，为每个时间步收集其前 block_size 个 token 的嵌入。
        for k in range(self.config.block_size):
            # 获取当前 idx 张量对应的 token 嵌入
            tok_emb = self.wte(idx)  # (B, T, n_embd)

            # 将 idx 向右滚动一个位置，为下一次迭代准备上下文。
            idx = torch.roll(idx, shifts=1, dims=1)
            # 将每一行滚到最前面的那个 token 设置为特殊的 <BLANK> token。
            idx[:, 0] = self.config.vocab_size

            embs.append(tok_emb)

        # 将所有收集到的嵌入向量在最后一个维度上拼接起来。
        # (B, T, n_embd) -> (B, T, n_embd * block_size)
        x = torch.cat(embs, dim=-1)

        # 将拼接后的向量输入 MLP，得到 logits。
        logits = self.mlp(x)  # (B, T, vocab_size)

        # 计算损失（如果提供了 targets）
        # 我们利用基类中定义的 _calculate_loss 辅助函数。
        loss = None
        if targets is not None:
            loss = self._calculate_loss(logits, targets)

        return logits, loss
