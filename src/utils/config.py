import yaml
import logging
from pathlib import Path
from .log import setup_logging


# 配置日志记录
setup_logging()
logger = logging.getLogger(__name__)


def load_config(
    config_path: str, base_config_path: str = "configs/base_config.yaml"
) -> dict:
    """加载并合并配置文件"""
    # 加载基础配置
    base_config = {}
    if Path(base_config_path).exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

    # 加载用户配置
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)

    # 合并配置
    return merge_configs(base_config, user_config)


def merge_configs(base_config: dict, user_config: dict) -> dict:
    """深度合并配置字典"""
    merged = base_config.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
