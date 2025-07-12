# src/data/__init__.py

from .datamodule import CharDataModule
from .dataset import CharDataset

# 这使得当其他文件导入 `data` 包时，这两个类是可用的。
__all__ = ["CharDataModule", "CharDataset"]
