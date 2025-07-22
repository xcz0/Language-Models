# src/models/base.py

import abc
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_embd2: int = 64


@dataclass
class AdamConfig:
    """AdamW 优化器的配置类"""

    learning_rate: float = 0.0005  # 5e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8


class LitBaseModel(LightningModule, abc.ABC):
    """
    一个抽象基类，继承自 LightningModule，为本项目所有模型提供通用框架。

    它处理了：
    - 训练和验证步骤的通用逻辑。
    - 优化器的配置。
    - 损失计算的辅助函数。

    子类必须实现 `__init__` 和 `forward` 方法。
    """

    def __init__(
        self,
        config: ModelConfig,
        optim_config: Optional[AdamConfig] = None,
    ):
        """
        初始化基类。

        Args:
            config (ModelConfig): 包含模型架构参数的配置对象。
            optim_config (AdamConfig, optional): 优化器配置。如果为 None，则使用默认值。
        """
        super().__init__()
        # 将所有构造函数参数保存到 hparams 属性中，并使得它们可以被 Lightning 的回调函数（如 ModelCheckpoint）访问。
        self.save_hyperparameters()

        self.config = config
        self.optim_config = optim_config if optim_config is not None else AdamConfig()

    @abc.abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        模型的前向传播。子类必须实现此方法。

        Args:
            idx (torch.Tensor): 输入的 token 索引张量，形状为 (B, T)。
            targets (torch.Tensor, optional): 目标的 token 索引张量，形状为 (B, T)。如果提供，则应计算并返回损失。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - logits: 模型的输出，形状为 (B, T, vocab_size)。
            - loss: 如果提供了 targets，则为计算出的交叉熵损失；否则为 None。
        """
        pass

    def _calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算交叉熵损失的辅助函数。
        """
        # CrossEntropyLoss 要求 logits 的形状为 (N, C)，其中 C 是类别数。
        # targets 的形状为 (N)。
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def training_step(self, batch, batch_idx):
        """执行一个训练步骤。"""
        x, y = batch
        _, loss = self(x, targets=y)
        # self.log 会自动将损失记录到 TensorBoard 或其他日志记录器中。
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """执行一个验证步骤。"""
        x, y = batch
        _, loss = self(x, targets=y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """配置模型的优化器。"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_config.learning_rate,
            weight_decay=self.optim_config.weight_decay,
            betas=self.optim_config.betas,
            eps=self.optim_config.eps,
        )
        return optimizer

    def get_block_size(self) -> int:
        """返回模型的上下文窗口大小。"""
        return self.config.block_size
