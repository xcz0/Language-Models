import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..base import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力机制 (Causal Multi-Head Self-Attention)。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 将 Q, K, V 的计算合并到一个大的线性层中，以提高效率
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 因果掩码 (causal mask)
        # 注册为 buffer，这样它不会被视为模型参数，但会随模型移动（如 .to(device)）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        # 为了类型检查，显式声明 bias 的类型
        self.bias: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Time, Channels (n_embd)

        # 1. 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. 调整形状以支持多头
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. 计算注意力分数
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 4. 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax 和加权求和
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # 6. 重组多头输出并投影
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Decoder(nn.Module):
    """标准的 Transformer 解码器块。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 原作者使用了 GELU 的一个特定近似实现，这里我们直接使用 PyTorch 的
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接 + 层归一化
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
