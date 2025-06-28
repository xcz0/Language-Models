# data_pipeline/tokenization/bpe_tokenizer.py

import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    一个从头开始实现的 BPE (Byte-Pair Encoding) 分词器。
    """

    def __init__(self):
        """初始化BPE分词器"""
        super().__init__()
        # 从 token 字符串到整数 ID 的映射
        self.token_to_id: Dict[str, int] = {}
        # 从整数 ID 到 token 字符串的映射
        self.id_to_token: Dict[int, str] = {}
        # 合并规则，(tok1, tok2) -> rank。rank 越小，合并越早
        self.merges: Dict[Tuple[int, int], int] = {}
        # 用于预分词的正则表达式模式 (来自 GPT-2)
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        # 特殊 token
        self.special_tokens: Dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(
        self,
        text: Union[str, List[str]],
        vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        """从原始文本训练 BPE 分词器"""
        assert vocab_size >= 256
        if special_tokens is None:
            special_tokens = []

        # 使用基类方法验证和处理文本输入
        combined_text = self._validate_text_input(text)

        # 1. 预分词和初始化词汇表
        text_chunks = re.findall(self.pattern, combined_text)

        # 将文本块编码为 UTF-8 字节序列，然后拆分成单个字节的列表
        # word_splits: [['w', 'o', 'r', 'd'], ['a', 'n', 'o', 't', 'h', 'e', 'r']]
        word_splits = [[c for c in chunk.encode("utf-8")] for chunk in text_chunks]

        # 初始化词汇表为所有单个字节 (0-255)
        # BPE 的基础单元是字节，而不是字符
        vocab = {i: bytes([i]) for i in range(256)}

        # 添加特殊 tokens
        for i, token_str in enumerate(special_tokens):
            vocab[256 + i] = token_str.encode("utf-8")

        num_merges = vocab_size - len(vocab)

        # 2. 迭代学习合并规则
        print("Starting BPE training...")
        for i in range(num_merges):
            # 计算相邻 token 对的频率
            stats = defaultdict(int)
            for chunk in word_splits:
                for p1, p2 in zip(chunk[:-1], chunk[1:]):
                    stats[(p1, p2)] += 1

            if not stats:
                break  # 没有更多可合并的对了

            # 找到最频繁的对
            most_freq_pair = max(stats, key=lambda x: stats[x])

            # 创建新的 token
            new_token_id = 256 + len(self.merges) + len(special_tokens)

            # 合并
            word_splits = self._merge_pair(word_splits, most_freq_pair, new_token_id)

            # 存储合并规则和新 token
            self.merges[most_freq_pair] = i
            vocab[new_token_id] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

            if (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: {most_freq_pair} -> {new_token_id}")

        # 3. 构建最终的 token <-> id 映射
        self.token_to_id = {
            v.decode("utf-8", errors="replace"): k for k, v in vocab.items()
        }
        self.id_to_token = {
            k: v.decode("utf-8", errors="replace") for k, v in vocab.items()
        }

        # 设置训练完成标志
        self._fitted = True
        self.logger.info(f"BPE训练完成，词汇表大小: {self.vocab_size}")

    def _merge_pair(self, word_splits, pair, new_id):
        new_word_splits = []
        for chunk in word_splits:
            i = 0
            new_chunk = []
            while i < len(chunk):
                if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                    new_chunk.append(new_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_word_splits.append(new_chunk)
        return new_word_splits

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """对单个字节块进行编码"""
        tokens = [b for b in text_bytes]
        while True:
            stats = defaultdict(int)
            for p1, p2 in zip(tokens[:-1], tokens[1:]):
                stats[(p1, p2)] += 1

            if not stats:
                break  # 没有相邻的token对，退出

            # 找到在 self.merges 中 rank 最低的（即最早学习到的）pair
            best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if best_pair not in self.merges:
                break  # 没有更多可应用的合并规则

            new_id = self.merges[best_pair] + 256 + len(self.special_tokens)
            tokens = self._merge_pair([tokens], best_pair, new_id)[0]
        return tokens

    def encode(self, text: str, **kwargs) -> List[int]:
        """将字符串编码为 token ID 列表"""
        tokens = []
        for chunk in re.findall(self.pattern, text):
            chunk_bytes = chunk.encode("utf-8")
            chunk_tokens = self._encode_chunk(chunk_bytes)
            tokens.extend(chunk_tokens)
        return tokens

    def decode(self, tokens: List[int], **kwargs) -> str:
        """将 token ID 列表解码为字符串"""
        # 注意：这里需要从 id -> bytes -> str
        vocab_bytes = {k: v.encode("utf-8") for k, v in self.id_to_token.items()}
        text_bytes = b"".join(vocab_bytes[idx] for idx in tokens)
        return text_bytes.decode("utf-8", errors="replace")

    def _prepare_save_data(self) -> Dict:
        """准备要保存的数据"""
        base_data = super()._prepare_save_data()
        # 将 tuple 类型的 key 转换为字符串 "tok1,tok2"
        merges_to_save = {" ".join(map(str, k)): v for k, v in self.merges.items()}

        base_data.update(
            {
                "token_to_id": self.token_to_id,
                "merges": merges_to_save,
                "pattern": self.pattern.pattern,
            }
        )
        return base_data

    def _validate_loaded_data(self, data: Dict):
        """验证加载的数据"""
        super()._validate_loaded_data(data)
        required_fields = ["token_to_id", "merges", "pattern"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"文件缺少必要字段: {field}")

    def _load_specific_data(self, data: Dict):
        """加载BPE分词器特定的数据"""
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # 将字符串 "tok1,tok2" key 转换回 tuple
        merge_dict = {}
        for k, v in data["merges"].items():
            parts = list(map(int, k.split()))
            if len(parts) == 2:
                merge_dict[(parts[0], parts[1])] = int(v)
        self.merges = merge_dict
        self.pattern = re.compile(data["pattern"])
