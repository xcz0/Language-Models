# data_pipeline/__init__.py

# 暴露顶层接口，方便用户使用
from .processing import create_data_processor
from .cleaners import create_text_cleaner
from .tokenization import CharTokenizer

# 配置日志
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
