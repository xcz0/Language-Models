# src/lit_language_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class LitLanguageModel(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # 保存超参数，以便从 checkpoint 加载
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch):
        x, y = batch
        logits = self(x)  # 调用 forward 方法

        # CrossEntropyLoss 要求 logits 的形状是 (N, C, ...) 和 target 的形状是 (N, ...)
        # 我们的 logits 是 (B, T, V) -> (Batch, Time, Vocab)
        # 我们的 y 是 (B, T)
        # 所以我们需要将它们 reshape
        B, T, V = logits.shape
        logits_reshaped = logits.view(B * T, V)
        y_reshaped = y.view(B * T)

        loss = F.cross_entropy(logits_reshaped, y_reshaped)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 使用 AdamW 是一个很好的默认选择
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
