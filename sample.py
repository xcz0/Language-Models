# src/sample.py

import torch
import argparse
import os
import pickle
import torch.nn.functional as F

from src.models import get_model_class


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    从模型生成序列的辅助函数。
    与原始脚本中的 `generate` 函数几乎完全相同。
    """
    model.eval()
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main(args):
    """主采样函数"""

    # 1. 确定并加载词汇表
    # 假设 vocab.pkl 与 checkpoint 在同一目录下或在其父目录中
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    vocab_path = os.path.join(checkpoint_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        # 尝试在父目录查找
        vocab_path = os.path.join(os.path.dirname(checkpoint_dir), "vocab.pkl")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Could not find vocab.pkl in {checkpoint_dir} or its parent."
            )

    with open(vocab_path, "rb") as f:
        meta_info = pickle.load(f)

    chars = meta_info["chars"]
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    # 添加特殊标记的解码
    itos[0] = ""  # <START> 或 <STOP> 标记解码为空字符串

    # 2. 从检查点加载模型
    # 首先，我们需要知道模型的类型。它存储在 hparams 中。
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_type = (
        checkpoint["hyper_parameters"]["config"]
        .__dict__["__dataclass_fields__"]["type"]
        .default
    )  # 有点曲折，但可以工作

    # 另一种更简单的方式是假设检查点文件所在的目录名就是模型类型
    # model_type = os.path.basename(os.path.dirname(args.checkpoint_path))

    model_class = get_model_class(model_type)
    model = model_class.load_from_checkpoint(args.checkpoint_path)
    model.to(args.device)
    model.eval()  # 设置为评估模式

    print(f"Model loaded from {args.checkpoint_path}")
    print(f"Model #params: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 生成样本
    print("-" * 80)
    print(f"Generating {args.num_samples} samples...")

    # 起始标记 <START> (索引为 0)
    start_token = torch.zeros(args.num_samples, 1, dtype=torch.long, device=args.device)

    # 生成索引序列
    max_len = model.get_block_size() - 1  # -1 for the start token
    generated_indices = generate(
        model, start_token, max_len, temperature=args.temperature, top_k=args.top_k
    )

    # 4. 解码并打印结果
    for i in range(generated_indices.size(0)):
        row = generated_indices[i, 1:].tolist()  # 忽略起始的 <START> 标记

        # 截断到 <STOP> 标记 (索引 0)
        if 0 in row:
            row = row[: row.index(0)]

        word_samp = "".join(itos[j] for j in row)
        print(word_samp)

    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a trained model.")

    # 输入
    parser.add_argument(
        "--checkpoint-path",
        "-c",
        type=str,
        required=True,
        help="Path to the model .ckpt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for sampling (e.g., 'cpu', 'cuda').",
    )

    # 采样参数
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. >1.0 for more randomness, <1.0 for less.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling. If specified, only sample from the top k most likely tokens.",
    )

    args = parser.parse_args()
    main(args)
