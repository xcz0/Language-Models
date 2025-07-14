# src/models/__init__.py

from typing import Type

from .base import ModelConfig
from .base import LitBaseModel
from .bigram import Bigram
# 当添加新模型时，只需在这里导入它们
# from .mlp import MLP
# from .transformer import Transformer

# 定义一个模型注册表（字典）
MODELS = {
    "bigram": Bigram,
    # "mlp": MLP,
    # "transformer": Transformer,
}


def get_model_class(model_name: str) -> Type[LitBaseModel]:
    """
    根据模型名称字符串获取模型类。

    Args:
        model_name (str): 模型的名称（例如 "bigram", "transformer"）。

    Returns:
        Type[LitBaseModel]: 对应的模型类。

    Raises:
        ValueError: 如果提供的模型名称在注册表中不存在。
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models are: {list(MODELS.keys())}"
        )
    return MODELS[model_name]


# 定义包的公共接口
__all__ = ["LitBaseModel", "get_model_class", "MODELS", "ModelConfig"]
