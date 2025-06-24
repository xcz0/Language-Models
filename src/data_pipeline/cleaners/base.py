# data_pipeline/cleaners/base.py

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTextCleaner(ABC):
    """
    文本清洗基类 - 定义通用的文本清洗接口。
    所有具体的清洗器都应继承此类。
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本的抽象方法。

        Args:
            text: 原始文本。

        Returns:
            清洗后的文本。
        """
        pass

    def extract_and_clean(
        self, data: List[Dict[str, Any]], content_field: str
    ) -> List[str]:
        """
        从原始数据列表中提取、清洗并返回内容列表。

        Args:
            data: 原始数据列表，每个元素是字典。
            content_field: 包含文本内容的字典键名。

        Returns:
            清洗后的文本内容列表。
        """
        cleaned_contents = []
        for item in data:
            content = item.get(content_field, "")
            if not content:
                continue

            cleaned_content = self.clean_text(content)
            if cleaned_content:
                cleaned_contents.append(cleaned_content)

        self.logger.info(
            f"从 {len(data)} 条记录中提取并清洗了 {len(cleaned_contents)} 条内容。"
        )
        return cleaned_contents

    def combine_texts(self, texts: List[str], separator: str = "\n") -> str:
        """
        将多个文本合并为一个大文本。

        Args:
            texts: 文本列表。
            separator: 分隔符。

        Returns:
            合并后的文本。
        """
        combined = separator.join(texts)
        self.logger.info(f"合并文本完成，总长度: {len(combined)} 字符。")
        return combined

    def get_text_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        获取文本统计信息。

        Args:
            texts: 文本列表。

        Returns:
            统计信息字典。
        """
        if not texts:
            return {}

        total_chars = sum(len(text) for text in texts)
        unique_chars = set("".join(texts))

        stats = {
            "总文本数量": len(texts),
            "总字符数": total_chars,
            "平均长度": round(total_chars / len(texts), 2) if texts else 0,
            "最大长度": max(len(text) for text in texts) if texts else 0,
            "最小长度": min(len(text) for text in texts) if texts else 0,
            "唯一字符数": len(unique_chars),
        }
        return stats
