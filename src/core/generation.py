# src/core/generation.py

import torch
import logging
from typing import Optional

from src.lit_language_model import LitLanguageModel
from src.tokenization.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_text(
    model: LitLanguageModel,
    tokenizer: BaseTokenizer,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: Optional[int],
    device: str,
) -> str:
    """核心文本生成函数。"""
    model.to(device)
    model.eval()

    tokens = (
        tokenizer.encode(prompt)
        if prompt
        else [int(torch.randint(0, tokenizer.vocab_size, (1,)).item())]
    )
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_length):
        logits = model(input_ids)[:, -1, :]  # (1, vocab_size)

        if temperature > 0:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:  # greedy decoding
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        tokens.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(tokens)
