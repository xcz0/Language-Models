"""
Transformer 语言模型的 Lightning 实现，结构与 GPT-2 完全一致。
"""

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn

from .base import LitBaseModel, ModelConfig, AdamConfig
from .layers.Attention import Decoder


class GPT(LitBaseModel):
    """
    仅解码器的 Transformer 语言模型。
    """

    def __init__(
        self,
        config: ModelConfig,
        optim_config: Optional[AdamConfig] = None,
    ):
        """
        初始化 Transformer 模型。

        Args:
            config (ModelConfig): 模型配置对象。
            optim_config (AdamConfig, optional): 优化器配置。
        """
        super().__init__(config, optim_config)

        # 嵌入层和 Transformer 块
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
                h=nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # 解码头
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # 权重共享：将解码头和词嵌入层的权重绑定在一起，这是 GPT-2 的标准做法
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"序列长度 {t} 超出 block_size {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # 1. 嵌入
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = tok_emb + pos_emb

        # 2. 通过所有 Transformer 块
        h_layers = cast(nn.ModuleList, self.transformer["h"])
        for block in h_layers:
            x = block(x)

        # 3. 最终的层归一化
        x = self.transformer["ln_f"](x)

        # 4. 解码
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = self._calculate_loss(logits, targets)

        return logits, loss
