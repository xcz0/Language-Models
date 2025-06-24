# src/data_processor.py

import json
import torch
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
import re

from .tokenizer import CharTokenizer


class PoetryDataProcessor:
    """
    诗歌数据处理器 - 负责数据清洗、预处理和格式化
    """

    def __init__(self, data_path: str, tokenizer_path: str):
        """
        初始化数据处理器

        Args:
            data_path: 原始数据文件路径
            tokenizer_path: 分词器保存路径
        """
        self.data_path = Path(data_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = CharTokenizer()
        self.logger = logging.getLogger(__name__)

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        加载原始JSON数据

        Returns:
            原始数据列表

        Raises:
            FileNotFoundError: 当数据文件不存在时
            json.JSONDecodeError: 当JSON格式错误时
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"成功加载 {len(data)} 条原始数据")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        清洗单个文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 1. 移除多余的空白字符
        text = re.sub(r"\s+", " ", text.strip())

        # 2. 移除特殊控制字符（保留换行符）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 3. 标准化标点符号
        text = text.replace("，", "，")  # 统一逗号
        text = text.replace("。", "。")  # 统一句号
        text = text.replace("？", "？")  # 统一问号
        text = text.replace("！", "！")  # 统一感叹号

        # 4. 移除重复的标点
        text = re.sub(r"([，。？！]){2,}", r"\1", text)

        return text.strip()

    def extract_poetry_content(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        从原始数据中提取并清洗诗歌内容

        Args:
            data: 原始数据列表

        Returns:
            清洗后的诗歌内容列表
        """
        poetry_contents = []

        for item in data:
            # 提取contents字段
            content = item.get("contents", "")
            if not content:
                continue

            # 清洗文本
            cleaned_content = self.clean_text(content)
            if cleaned_content:  # 只保留非空内容
                poetry_contents.append(cleaned_content)

        self.logger.info(f"提取并清洗了 {len(poetry_contents)} 首诗歌")
        return poetry_contents

    def combine_texts(self, texts: List[str], separator: str = "\n") -> str:
        """
        将多个文本合并为一个大文本

        Args:
            texts: 文本列表
            separator: 分隔符

        Returns:
            合并后的文本
        """
        combined = separator.join(texts)
        self.logger.info(f"合并文本完成，总长度: {len(combined)} 字符")
        return combined

    def build_and_save_tokenizer(self, text: str) -> CharTokenizer:
        """
        构建并保存分词器

        Args:
            text: 训练文本

        Returns:
            训练好的分词器
        """
        self.logger.info("开始构建分词器...")
        self.tokenizer.fit(text)

        # 创建保存目录
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存分词器
        self.tokenizer.save(self.tokenizer_path)
        self.logger.info(f"分词器已保存到: {self.tokenizer_path}")
        self.logger.info(f"词汇表大小: {self.tokenizer.vocab_size}")

        return self.tokenizer

    def load_tokenizer(self) -> CharTokenizer:
        """
        加载已训练的分词器

        Returns:
            加载的分词器

        Raises:
            FileNotFoundError: 当分词器文件不存在时
        """
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"分词器文件不存在: {self.tokenizer_path}")

        self.tokenizer.load(self.tokenizer_path)
        self.logger.info(f"分词器已加载，词汇表大小: {self.tokenizer.vocab_size}")
        return self.tokenizer

    def encode_text(self, text: str) -> torch.Tensor:
        """
        将文本编码为张量

        Args:
            text: 输入文本

        Returns:
            编码后的张量
        """
        if not self.tokenizer._fitted:
            raise ValueError("分词器尚未训练或加载")

        encoded = self.tokenizer.encode(text)
        tensor = torch.tensor(encoded, dtype=torch.long)
        self.logger.info(f"文本编码完成，张量形状: {tensor.shape}")
        return tensor

    def create_sequences(
        self, encoded_data: torch.Tensor, context_window: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建训练序列对 (输入, 目标)

        Args:
            encoded_data: 编码后的数据张量
            context_window: 上下文窗口大小

        Returns:
            (输入张量, 目标张量) 的元组
        """
        if len(encoded_data) <= context_window:
            raise ValueError(
                f"数据长度 {len(encoded_data)} 必须大于上下文窗口 {context_window}"
            )

        num_sequences = len(encoded_data) - context_window

        # 创建输入序列 (x)
        X = torch.stack(
            [encoded_data[i : i + context_window] for i in range(num_sequences)]
        )

        # 创建目标序列 (y) - 输入序列向后移动一位
        Y = torch.stack(
            [encoded_data[i + 1 : i + context_window + 1] for i in range(num_sequences)]
        )

        self.logger.info(f"创建了 {num_sequences} 个序列对")
        self.logger.info(f"输入张量形状: {X.shape}, 目标张量形状: {Y.shape}")

        return X, Y

    def process_all(
        self, context_window: int
    ) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
        """
        执行完整的数据处理流程

        Args:
            context_window: 上下文窗口大小

        Returns:
            (输入张量, 目标张量, 分词器) 的元组
        """
        # 1. 加载原始数据
        raw_data = self.load_raw_data()

        # 2. 提取并清洗诗歌内容
        poetry_contents = self.extract_poetry_content(raw_data)

        # 3. 合并文本
        full_text = self.combine_texts(poetry_contents)

        # 4. 构建并保存分词器
        tokenizer = self.build_and_save_tokenizer(full_text)

        # 5. 编码文本
        encoded_data = self.encode_text(full_text)

        # 6. 创建序列对
        X, Y = self.create_sequences(encoded_data, context_window)

        return X, Y, tokenizer

    def get_data_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        获取数据统计信息

        Args:
            texts: 文本列表

        Returns:
            统计信息字典
        """
        if not texts:
            return {}

        total_chars = sum(len(text) for text in texts)
        avg_length = total_chars / len(texts)
        max_length = max(len(text) for text in texts)
        min_length = min(len(text) for text in texts)

        # 统计字符种类
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        stats = {
            "总诗歌数量": len(texts),
            "总字符数": total_chars,
            "平均长度": round(avg_length, 2),
            "最大长度": max_length,
            "最小长度": min_length,
            "唯一字符数": len(unique_chars),
        }

        return stats


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建数据处理器
    processor = PoetryDataProcessor(
        data_path="/home/xcz/Language-Models/data/poetry.json",
        tokenizer_path="/home/xcz/Language-Models/data/tokenizer.pkl",
    )

    # 执行完整处理流程
    try:
        X, Y, tokenizer = processor.process_all(context_window=32)
        print(f"处理完成！输入形状: {X.shape}, 目标形状: {Y.shape}")
        print(f"词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"处理失败: {e}")
