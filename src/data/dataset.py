# src/data/dataset.py

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    一个 PyTorch 数据集，用于处理字符级语言模型任务。
    它接收一个单词列表，并将每个单词转换为一个用于训练的 (x, y) 对。

    x: 输入序列，以 <START> 标记 (0) 开头。
    y: 目标序列，是输入序列向左移动一位的结果，用于预测下一个字符。
       在序列末尾使用 -1 进行填充，以便在计算损失时忽略这些位置。
    """

    def __init__(self, words: list[str], chars: list[str], max_word_length: int):
        """
        初始化数据集。

        Args:
            words (List[str]): 数据集中的单词列表。
            chars (List[str]): 词汇表中的所有唯一字符。
            max_word_length (int): 数据集中最长单词的长度。
        """
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length

        # 创建字符到索引和索引到字符的映射
        # 索引 0 保留给 <START>/<STOP> 标记
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self) -> int:
        """返回数据集中的样本数量。"""
        return len(self.words)

    def contains(self, word: str) -> bool:
        """检查数据集中是否包含某个单词。"""
        return word in self.words

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小（包括特殊标记0）。"""
        return len(self.chars) + 1

    @property
    def output_length(self) -> int:
        """返回模型的输出序列长度（包含<START>标记）。"""
        return self.max_word_length + 1

    def encode(self, word: str) -> list[int]:
        """将一个字符串编码为索引列表。"""
        return [self.stoi[w] for w in word]

    def decode(self, ix: list[int]) -> str:
        """将一个索引列表解码为字符串。"""
        return "".join(self.itos[i] for i in ix)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集中的一个样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 一个包含输入张量 x 和目标张量 y 的元组。
        """
        word = self.words[idx]
        ix = self.encode(word)

        # 初始化输入和目标张量，长度为 max_word_length + 1
        x = torch.zeros(self.output_length, dtype=torch.long)
        y = torch.zeros(self.output_length, dtype=torch.long)

        # 填充输入张量 x，从第二个位置开始（第一个位置是 <START> 标记，值为0）
        x[1 : 1 + len(ix)] = torch.tensor(ix)

        # 填充目标张量 y，它是 x 的左移版本
        y[: len(ix)] = torch.tensor(ix)
        # 在目标序列的末尾添加 <STOP> 标记（值为0），并用 -1 填充其余部分
        # F.cross_entropy 会忽略索引为 -1 的目标
        y[len(ix)] = 0  # <STOP> token
        y[len(ix) + 1 :] = -1

        return x, y
