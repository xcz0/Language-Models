"""
BoW (Bag-of-Words) 语言模型的 Lightning 实现。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBaseModel, ModelConfig, AdamConfig


class CausalBoW(nn.Module):
    """
    因果词袋模块 (Causal Bag-of-Words)。

    这个模块实现了一种简单的上下文聚合机制。对于序列中的每个位置 T，它会计算所有从 0 到 T 的先前 token 特征的平均值。

    它通过一个巧妙的技巧实现这一点：创建一个因果掩码，然后应用 softmax，这会为所有非掩码位置（即过去和当前）分配相等的权重，从而实现平均。

    这可以被看作是一个权重固定的、简化的自注意力机制。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 预先计算并注册一个下三角矩阵作为缓冲区，用于实现因果掩码。
        # 这样可以确保注意力只施加到序列的左侧（过去）。
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, config.block_size, config.block_size
            ),
        )
        self.bias: torch.Tensor  # 添加类型注解

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dim

        # 创建一个注意力权重矩阵，初始为零
        att = torch.zeros((B, T, T), device=x.device)
        # 将未来的位置（由 bias 矩阵中的 0 指示）填充为负无穷
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float("-inf"))
        # 对最后一个维度应用 softmax。这使得所有过去和当前位置的权重相等，而未来位置的权重为0。
        att = F.softmax(att, dim=-1)

        # 执行加权平均。 (B, T, T) @ (B, T, C) -> (B, T, C)
        y = att @ x
        return y


class BoWBlock(nn.Module):
    """
    一个 BoW 块，它结合了 CausalBoW 和一个小型 MLP。
    使用了残差连接，这对于训练深度网络至关重要。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个残差连接：将输入的原始表示与经过词袋平均后的表示相加。
        x = x + self.cbow(x)
        # 第二个残差连接：将结果通过一个 MLP 进行变换，然后再次与输入相加。
        x = x + self.mlp(x)
        return x


class BoW(LitBaseModel):
    """
    一个基于词袋（Bag-of-Words）的语言模型。

    对于序列中的每个位置，该模型将其之前所有 token 的嵌入向量（包括位置嵌入）
    进行平均，然后使用这个平均后的上下文向量来预测下一个 token。
    """

    def __init__(
        self,
        config: ModelConfig,
        optim_config: Optional[AdamConfig] = None,
    ):
        """
        初始化 BoW 模型。

        Args:
            config (ModelConfig): 模型配置对象。
            optim_config (AdamConfig, optional): 优化器配置。
        """
        super().__init__(config, optim_config)

        # token 嵌入和位置嵌入
        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, self.config.n_embd)

        # 上下文处理块
        self.context_block = BoWBlock(config)

        # 最终的线性解码层（语言模型头）
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        BoW 模型的前向传播。
        """
        device = idx.device
        b, t = idx.size()

        # 确保序列长度不超过 block_size
        assert t <= self.config.block_size, (
            f"序列长度 {t} 超出 block_size {self.config.block_size}"
        )

        # 创建位置索引
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # 1. 嵌入层：获取 token 和 position 的嵌入向量
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (1, T, n_embd)

        # 将 token 嵌入和位置嵌入相加，为每个 token 注入位置信息
        x = tok_emb + pos_emb

        # 2. 上下文处理：通过 BoW 块进行特征提取
        x = self.context_block(x)

        # 3. 解码层：将最终的表示映射到词汇表的 logits
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = self._calculate_loss(logits, targets)

        return logits, loss
