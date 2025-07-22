"""
RNN (Recurrent Neural Network) 和 GRU (Gated Recurrent Unit) 语言模型
"""

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBaseModel, ModelConfig, AdamConfig


class RNNCell(nn.Module):
    """
    一个基础的 RNN 单元。它接收当前时间步的输入 `xt` 和前一时间步的隐藏状态 `h_prev`，并计算出当前时间步的隐藏状态 `h_next`。
    公式: h_next = tanh(W_h * [xt, h_prev] + b_h)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 线性层将拼接后的输入和前序隐藏状态映射到新的隐藏状态维度
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # 将输入 xt 和前序隐藏状态 h_prev 在维度 1 上拼接
        xh = torch.cat([xt, h_prev], dim=1)
        # 通过线性和 tanh 激活函数计算新的隐藏状态
        h_next = F.tanh(self.xh_to_h(xh))
        return h_next


class GRUCell(nn.Module):
    """
    一个门控循环单元 (GRU)。
    它通过重置门 (reset gate) 和更新门 (update gate) 来更复杂地控制信息流动，有助于缓解梯度消失问题，并捕获更长期的依赖。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # 更新门 (z_t): 决定在多大程度上保留旧的隐藏状态
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        # 重置门 (r_t): 决定忽略多少过去的信息
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        # 候选隐藏状态 (h_tilde_t)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([xt, h_prev], dim=1)

        # 1. 计算重置门 r_t，并用它来“重置”部分旧隐藏状态
        r = torch.sigmoid(self.xh_to_r(xh))
        h_prev_reset = r * h_prev

        # 2. 使用重置后的隐藏状态计算候选隐藏状态 h_tilde_t
        xhr = torch.cat([xt, h_prev_reset], dim=1)
        hbar = torch.tanh(self.xh_to_hbar(xhr))

        # 3. 计算更新门 z_t
        z = torch.sigmoid(self.xh_to_z(xh))

        # 4. 结合旧的隐藏状态和候选隐藏状态，生成新的隐藏状态 h_next
        # z 决定了新旧信息融合的比例
        h_next = (1 - z) * h_prev + z * hbar
        return h_next


class RNN(LitBaseModel):
    """
    一个基于循环神经网络的语言模型，支持 'rnn' 和 'gru' 两种单元类型。
    """

    def __init__(
        self,
        config: ModelConfig,
        cell_type: Literal["rnn", "gru"] = "gru",
        optim_config: Optional[AdamConfig] = None,
    ):
        super().__init__(config, optim_config)
        self.cell_type = cell_type

        # 词嵌入表
        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)

        # 可学习的初始隐藏状态 (h_0)
        self.start_hidden = nn.Parameter(torch.zeros(1, self.config.n_embd2))

        # 根据 cell_type 选择 RNN 单元
        if self.cell_type == "rnn":
            self.cell = RNNCell(config)
        elif self.cell_type == "gru":
            self.cell = GRUCell(config)
        else:
            raise ValueError(f"不支持的 cell_type: {self.cell_type}")

        # 解码层
        self.lm_head = nn.Linear(self.config.n_embd2, self.config.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()

        # 1. 获取所有时间步的词嵌入
        emb = self.wte(idx)  # (B, T, n_embd)

        # 2. 初始化隐藏状态
        # 将可学习的初始隐藏状态扩展到当前批次的大小
        h_prev = self.start_hidden.expand(b, -1)

        # 3. 循环遍历时间步
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]  # 获取当前时间步的输入 (B, n_embd)
            h_next = self.cell(xt, h_prev)  # 计算新的隐藏状态 (B, n_embd2)
            hiddens.append(h_next)
            h_prev = h_next  # 更新隐藏状态以备下一时间步使用

        # 4. 收集所有隐藏状态并解码
        hidden = torch.stack(hiddens, dim=1)  # (B, T, n_embd2)
        logits = self.lm_head(hidden)

        # 计算损失
        loss = None
        if targets is not None:
            loss = self._calculate_loss(logits, targets)

        return logits, loss
