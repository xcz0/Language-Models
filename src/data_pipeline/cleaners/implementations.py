# data_pipeline/cleaners/implementations.py

import re
from typing import List, Dict, Optional
from .base import BaseTextCleaner


class PoetryTextCleaner(BaseTextCleaner):
    """诗歌文本清洗器 - 专门用于清洗古诗文数据。"""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = (
            text.replace("，", "，")
            .replace("。", "。")
            .replace("？", "？")
            .replace("！", "！")
        )
        text = re.sub(r"([，。？！]){2,}", r"\1", text)
        text = re.sub(r"\[.*?\]|（.*?）|\(.*?\)", "", text)
        text = re.sub(r"第\d+页|卷\d+", "", text)
        return text.strip()


class GeneralTextCleaner(BaseTextCleaner):
    """通用文本清洗器 - 用于一般文本数据的清洗。"""

    def __init__(self, preserve_structure: bool = True):
        super().__init__()
        self.preserve_structure = preserve_structure

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        if self.preserve_structure:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n", "\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = (
            text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        )
        return text.strip()


class CustomTextCleaner(BaseTextCleaner):
    """自定义文本清洗器 - 允许用户定义清洗规则。"""

    def __init__(
        self,
        custom_patterns: Optional[List[Dict[str, str]]] = None,
        custom_replacements: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.custom_patterns = custom_patterns or []
        self.custom_replacements = custom_replacements or {}

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = re.sub(r"\s+", " ", text.strip())
        for pattern_dict in self.custom_patterns:
            pattern = pattern_dict.get("pattern", "")
            replacement = pattern_dict.get("replacement", "")
            if pattern:
                text = re.sub(pattern, replacement, text)
        for old, new in self.custom_replacements.items():
            text = text.replace(old, new)
        return text.strip()
