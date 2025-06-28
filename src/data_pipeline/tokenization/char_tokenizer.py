# data_pipeline/tokenization/char_tokenizer.py

from typing import List, Dict, Optional, Union
import warnings

from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
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
        super().__init__()
        self.char_to_token: Dict[str, int] = {}
        self.token_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []  # 改为list以保持顺序

        # 特殊token设置
        self.add_special_tokens = add_special_tokens
        self.unk_token = unk_token
        self.pad_token = pad_token

    def _add_special_tokens(self):
        """添加特殊token到词汇表"""
        special_tokens = []
        if self.add_special_tokens:
            special_tokens = [self.pad_token, self.unk_token]
        return special_tokens

    def fit(self, text: Union[str, List[str]], **kwargs):
        """
        从文本构建词汇表

        Args:
            text: 训练文本，可以是字符串或字符串列表

        Raises:
            ValueError: 当输入文本为空时
        """
        # 处理多种输入类型
        combined_text = self._validate_text_input(text)

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

    def encode(self, text: str, handle_unk: bool = True, **kwargs) -> List[int]:
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
        self._validate_fitted()

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

    def decode(
        self, tokens: List[int], skip_special_tokens: bool = False, **kwargs
    ) -> str:
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
        self._validate_fitted()

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

    def _prepare_save_data(self) -> Dict[str, Union[str, int, bool, List, Dict]]:
        """准备要保存的数据"""
        base_data = super()._prepare_save_data()
        base_data.update(
            {
                "char_to_token": self.char_to_token,
                "token_to_char": self.token_to_char,
                "vocab": self.vocab,
                "add_special_tokens": self.add_special_tokens,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
            }
        )
        return base_data

    def _validate_loaded_data(self, data: Dict):
        """验证加载的数据"""
        super()._validate_loaded_data(data)
        required_fields = ["char_to_token", "token_to_char", "vocab"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"文件缺少必要字段: {field}")

    def _load_specific_data(self, data: Dict):
        """加载字符分词器特定的数据"""
        self.char_to_token = data["char_to_token"]

        # 处理token_to_char的key类型（JSON格式的key会是字符串）
        if isinstance(list(data["token_to_char"].keys())[0], str):
            # JSON格式，需要转换key为整数
            self.token_to_char = {int(k): v for k, v in data["token_to_char"].items()}
        else:
            # pickle格式，key已经是整数
            self.token_to_char = data["token_to_char"]

        self.vocab = data["vocab"]

        # 加载配置（向后兼容）
        self.add_special_tokens = data.get("add_special_tokens", True)
        self.unk_token = data.get("unk_token", self.UNK_TOKEN)
        self.pad_token = data.get("pad_token", self.PAD_TOKEN)

    def get_vocab(self) -> Dict[str, int]:
        """
        返回词汇表字典

        Returns:
            字符到token ID的映射
        """
        return self.char_to_token.copy()
