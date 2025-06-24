# data_pipeline/cleaners/factory.py

from .base import BaseTextCleaner
from .implementations import PoetryTextCleaner, GeneralTextCleaner, CustomTextCleaner


def create_text_cleaner(cleaner_type: str = "poetry", **kwargs) -> BaseTextCleaner:
    """
    文本清洗器工厂函数。

    Args:
        cleaner_type: 清洗器类型 ("poetry", "general", "custom")。
        **kwargs: 传递给清洗器构造函数的参数。

    Returns:
        对应的文本清洗器实例。
    """
    cleaner_map = {
        "poetry": PoetryTextCleaner,
        "general": GeneralTextCleaner,
        "custom": CustomTextCleaner,
    }
    cleaner_class = cleaner_map.get(cleaner_type)
    if not cleaner_class:
        raise ValueError(f"不支持的清洗器类型: {cleaner_type}")

    # 过滤掉不适用于特定清洗器的kwargs，以避免TypeError
    import inspect

    sig = inspect.signature(cleaner_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return cleaner_class(**valid_kwargs)
