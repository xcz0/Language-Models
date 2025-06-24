# src/tokenizer_optimized.py

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings


class CharTokenizer:
    """
    字符级分词器 (Character-level Tokenizer)
    """

    # 特殊token常量
    UNK_TOKEN = "<UNK>"  # 未知字符token
    PAD_TOKEN = "<PAD>"  # 填充token

    def __init__(
        self,
        add_special_tokens: bool = True,
        unk_token: str = UNK_TOKEN,
        pad_token: str = PAD_TOKEN,
    ):
        """
        初始化分词器

        Args:
            add_special_tokens: 是否添加特殊token
            unk_token: 未知字符token
            pad_token: 填充token
        """
        self.char_to_token: Dict[str, int] = {}
        self.token_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []  # 改为list以保持顺序
        self._fitted = False

        # 特殊token设置
        self.add_special_tokens = add_special_tokens
        self.unk_token = unk_token
        self.pad_token = pad_token

        # 配置日志
        self.logger = logging.getLogger(__name__)

    def _add_special_tokens(self):
        """添加特殊token到词汇表"""
        special_tokens = []
        if self.add_special_tokens:
            special_tokens = [self.pad_token, self.unk_token]
        return special_tokens

    def fit(self, text: Union[str, List[str]]):
        """
        从文本构建词汇表

        Args:
            text: 训练文本，可以是字符串或字符串列表

        Raises:
            ValueError: 当输入文本为空时
        """
        if not text:
            raise ValueError("输入文本不能为空")

        # 处理多种输入类型
        if isinstance(text, list):
            combined_text = "".join(text)
        else:
            combined_text = text

        if not combined_text:
            raise ValueError("文本内容不能为空")

        # 构建词汇表：特殊token + 排序的唯一字符
        special_tokens = self._add_special_tokens()
        unique_chars = sorted(list(set(combined_text)))

        # 确保特殊token不会重复
        for token in special_tokens:
            if token in unique_chars:
                unique_chars.remove(token)

        self.vocab = special_tokens + unique_chars

        # 构建映射关系
        self.char_to_token = {ch: i for i, ch in enumerate(self.vocab)}
        self.token_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        self._fitted = True
        self.logger.info(f"词汇表构建完成，包含 {len(self.vocab)} 个token")

    def encode(self, text: str, handle_unk: bool = True) -> List[int]:
        """
        将字符串编码为整数列表

        Args:
            text: 要编码的文本
            handle_unk: 是否处理未知字符

        Returns:
            编码后的token ID列表

        Raises:
            ValueError: 当分词器未训练时
            KeyError: 当存在未知字符且未启用处理时
        """
        if not self._fitted:
            raise ValueError("分词器尚未训练，请先调用 fit() 方法")

        if not text:
            return []

        tokens = []
        unk_id = self.char_to_token.get(self.unk_token)

        for char in text:
            if char in self.char_to_token:
                tokens.append(self.char_to_token[char])
            elif handle_unk and unk_id is not None:
                tokens.append(unk_id)
                warnings.warn(f"遇到未知字符: '{char}'，已替换为 {self.unk_token}")
            else:
                raise KeyError(f"字符 '{char}' 不在词汇表中")

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """
        将整数列表解码为字符串

        Args:
            tokens: token ID列表
            skip_special_tokens: 是否跳过特殊token

        Returns:
            解码后的字符串

        Raises:
            ValueError: 当分词器未训练时
            KeyError: 当token ID不存在时
        """
        if not self._fitted:
            raise ValueError("分词器尚未训练，请先调用 fit() 方法")

        if not tokens:
            return ""

        chars = []
        special_tokens = (
            {self.pad_token, self.unk_token} if self.add_special_tokens else set()
        )

        for token in tokens:
            if token not in self.token_to_char:
                raise KeyError(f"Token ID {token} 不在词汇表中")

            char = self.token_to_char[token]
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)

        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> Optional[int]:
        """返回填充token的ID"""
        return self.char_to_token.get(self.pad_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """返回未知token的ID"""
        return self.char_to_token.get(self.unk_token)

    def save(self, filepath: Union[str, Path], format: str = "pickle"):
        """
        将词汇表保存到文件

        Args:
            filepath: 保存路径
            format: 保存格式，可选 "pickle" 或 "json"

        Raises:
            ValueError: 当分词器未训练时或格式不支持时
            OSError: 当文件操作失败时
        """
        if not self._fitted:
            raise ValueError("分词器尚未训练，无法保存")

        if format not in ["pickle", "json"]:
            raise ValueError("格式必须是 'pickle' 或 'json'")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        vocab_to_save = {
            "char_to_token": self.char_to_token,
            "token_to_char": self.token_to_char,
            "vocab": self.vocab,
            "add_special_tokens": self.add_special_tokens,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "version": "1.0",
        }

        try:
            if format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(vocab_to_save, f)
                self.logger.info(f"分词器已保存到 {filepath} (pickle格式)")
            else:  # json format
                # 对于JSON格式，需要转换token_to_char的key为字符串
                vocab_to_save["token_to_char"] = {
                    str(k): v for k, v in vocab_to_save["token_to_char"].items()
                }
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
                self.logger.info(f"分词器已保存到 {filepath} (JSON格式)")
        except OSError as e:
            self.logger.error(f"保存失败: {e}")
            raise

    def load(self, filepath: Union[str, Path]):
        """
        从文件加载词汇表，自动检测文件格式

        Args:
            filepath: 文件路径

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式不正确时
            OSError: 当文件操作失败时
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 自动检测文件格式
        try:
            # 首先尝试加载为pickle文件
            with open(filepath, "rb") as f:
                vocab_loaded = pickle.load(f)
            self.logger.info("检测到pickle格式文件")
        except (pickle.UnpicklingError, UnicodeDecodeError):
            # 如果pickle失败，尝试JSON格式
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    vocab_loaded = json.load(f)
                self.logger.info("检测到JSON格式文件")
            except json.JSONDecodeError as e:
                raise ValueError(f"文件格式错误，既不是有效的pickle也不是JSON: {e}")
        except OSError as e:
            self.logger.error(f"加载失败: {e}")
            raise

        # 验证必要字段
        required_fields = ["char_to_token", "token_to_char", "vocab"]
        for field in required_fields:
            if field not in vocab_loaded:
                raise ValueError(f"文件缺少必要字段: {field}")

        # 加载数据
        self.char_to_token = vocab_loaded["char_to_token"]
        # 处理token_to_char的key类型（JSON格式的key会是字符串）
        if isinstance(list(vocab_loaded["token_to_char"].keys())[0], str):
            # JSON格式，需要转换key为整数
            self.token_to_char = {
                int(k): v for k, v in vocab_loaded["token_to_char"].items()
            }
        else:
            # pickle格式，key已经是整数
            self.token_to_char = vocab_loaded["token_to_char"]

        self.vocab = vocab_loaded["vocab"]

        # 加载配置（向后兼容）
        self.add_special_tokens = vocab_loaded.get("add_special_tokens", True)
        self.unk_token = vocab_loaded.get("unk_token", self.UNK_TOKEN)
        self.pad_token = vocab_loaded.get("pad_token", self.PAD_TOKEN)

        self._fitted = True
        self.logger.info(f"分词器已从 {filepath} 加载")

    def get_vocab(self) -> Dict[str, int]:
        """
        返回词汇表字典

        Returns:
            字符到token ID的映射
        """
        return self.char_to_token.copy()

    def __len__(self) -> int:
        """返回词汇表大小"""
        return self.vocab_size

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        status = "fitted" if self._fitted else "not fitted"
        return f"CharTokenizer(vocab_size={self.vocab_size}, status={status})"


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建分词器
    tokenizer = CharTokenizer()

    # 训练
    sample_text = "Hello, 世界! This is a test."
    tokenizer.fit(sample_text)

    # 编码和解码
    encoded = tokenizer.encode("Hello")
    decoded = tokenizer.decode(encoded)

    print("原文: Hello")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 保存和加载示例
    print("\n保存和加载示例:")

    # 保存为pickle格式（默认）
    tokenizer.save("tokenizer.pkl")

    # 保存为JSON格式（可选）
    tokenizer.save("tokenizer.json", format="json")

    # 创建新的分词器并加载
    new_tokenizer = CharTokenizer()
    new_tokenizer.load("tokenizer.pkl")  # 自动检测格式

    # 验证加载是否成功
    test_encoded = new_tokenizer.encode("Hello")
    print(f"加载后编码结果: {test_encoded}")
    print(f"编码结果一致: {encoded == test_encoded}")
