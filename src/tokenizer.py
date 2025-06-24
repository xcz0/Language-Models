# src/tokenizer.py

import json
from typing import List


class CharTokenizer:
    """
    字符级分词器 (Character-level Tokenizer)
    """

    def __init__(self):
        self.char_to_token = {}
        self.token_to_char = {}
        self.vocab = set()

    def fit(self, text: str):
        """从文本构建词汇表"""
        self.vocab = sorted(list(set(text)))
        self.char_to_token = {ch: i for i, ch in enumerate(self.vocab)}
        self.token_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """将字符串编码为整数列表"""
        return [self.char_to_token[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        """将整数列表解码为字符串"""
        return "".join([self.token_to_char[token] for token in tokens])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, filepath: str):
        """将词汇表保存到文件"""
        vocab_to_save = {
            "char_to_token": self.char_to_token,
            "token_to_char": self.token_to_char,
            "vocab": list(self.vocab),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer vocabulary saved to {filepath}")

    def load(self, filepath: str):
        """从文件加载词汇表"""
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_loaded = json.load(f)
        self.char_to_token = vocab_loaded["char_to_token"]
        # 注意: json加载后，key会变成字符串，需要转换回来
        self.token_to_char = {
            int(k): v for k, v in vocab_loaded["token_to_char"].items()
        }
        self.vocab = set(vocab_loaded["vocab"])
        print(f"Tokenizer vocabulary loaded from {filepath}")
