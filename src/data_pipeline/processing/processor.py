# data_pipeline/processing/processor.py

import logging
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from ..cleaners import create_text_cleaner, BaseTextCleaner
from ..tokenization import CharTokenizer
from ..utils import io
from . import sequencing


class DataProcessor:
    """
    数据处理协调器。
    负责编排数据加载、清洗、分词、序列化和保存的整个流程。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        processed_data_path: str,
        text_cleaner: BaseTextCleaner,
        content_field: str = "contents",
    ):
        self.data_path = Path(data_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.processed_data_path = Path(processed_data_path)
        self.text_cleaner = text_cleaner
        self.content_field = content_field
        self.tokenizer = CharTokenizer()
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_all(self, context_window: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行完整的数据处理流程，并保存结果。

        Args:
            context_window: 用于创建序列的上下文窗口大小。

        Returns:
            (输入张量, 目标张量)
        """
        # 1. 加载和清洗
        raw_data = io.load_json_data(self.data_path)
        cleaned_texts = self.text_cleaner.extract_and_clean(
            raw_data, self.content_field
        )
        full_text = self.text_cleaner.combine_texts(cleaned_texts)

        # 2. 分词
        self.tokenizer.fit(full_text)
        self.tokenizer.save(self.tokenizer_path)
        self.logger.info(
            f"分词器构建完成并保存到 {self.tokenizer_path}，词汇表大小: {self.tokenizer.vocab_size}"
        )

        # 3. 编码和序列化
        encoded_data = torch.tensor(self.tokenizer.encode(full_text), dtype=torch.long)
        X, Y = sequencing.create_sequences(encoded_data, context_window)

        # 4. 保存处理结果
        stats = self.text_cleaner.get_text_stats(cleaned_texts)
        metadata = {
            "processing_date": datetime.now().isoformat(),
            "original_data_stats": stats,
            "full_text_length": len(full_text),
        }
        processed_data_package = {
            "X": X,
            "Y": Y,
            "tokenizer": self.tokenizer,
            "metadata": metadata,
        }
        io.save_processed_data(self.processed_data_path, processed_data_package)

        return X, Y

    def get_or_process_data(
        self, context_window: int, force_reprocess: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
        """
        获取处理好的数据。如果已存在且上下文窗口匹配，则加载；否则重新处理。

        Args:
            context_window: 上下文窗口大小。
            force_reprocess: 是否强制重新处理。

        Returns:
            (输入张量, 目标张量, 分词器实例)
        """
        if not force_reprocess and self.processed_data_path.exists():
            self.logger.info("发现已处理的数据文件，尝试加载...")
            try:
                data = io.load_processed_data(self.processed_data_path)
                if data["X"].shape[1] == context_window:
                    self.logger.info("数据加载成功，上下文窗口匹配。")
                    return data["X"], data["Y"], data["tokenizer"]
                else:
                    self.logger.warning("上下文窗口不匹配，将重新处理数据。")
            except Exception as e:
                self.logger.warning(f"加载数据失败: {e}，将重新处理数据。")

        self.logger.info("开始处理原始数据...")
        X, Y = self.process_all(context_window)
        return X, Y, self.tokenizer


def create_data_processor(
    data_path: str,
    tokenizer_path: str,
    processed_data_path: str,
    cleaner_type: str = "poetry",
    **cleaner_kwargs,
) -> DataProcessor:
    """
    数据处理器工厂函数。

    Args:
        data_path: 原始数据文件路径。
        tokenizer_path: 分词器保存路径。
        processed_data_path: 处理后数据保存路径。
        cleaner_type: 文本清洗器类型。
        **cleaner_kwargs: 传递给文本清洗器的参数，如 content_field, custom_patterns 等。

    Returns:
        配置好的数据处理器实例。
    """
    text_cleaner = create_text_cleaner(cleaner_type, **cleaner_kwargs)
    content_field = cleaner_kwargs.get("content_field", "contents")

    return DataProcessor(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        processed_data_path=processed_data_path,
        text_cleaner=text_cleaner,
        content_field=content_field,
    )
