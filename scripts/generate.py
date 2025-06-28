# scripts/generate.py

import logging
import torch
from src.core.generation import generate_text
from src.utils.config import load_config
from src.utils.log import setup_logging
from src.utils.model_loader import load_tokenizer, load_model_from_checkpoint

logger = logging.getLogger(__name__)


def run_generation(
    config_path,
    checkpoint_path,
    tokenizer_path,
    prompt,
    max_length,
    temperature,
    top_k,
    num_samples,
    device,
):
    """编排批量文本生成任务。"""
    setup_logging()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, config, tokenizer.vocab_size)

    for i in range(num_samples):
        print(f"\n--- Sample {i + 1}/{num_samples} ---")
        output = generate_text(
            model, tokenizer, prompt, max_length, temperature, top_k, device
        )
        print(output)


def run_interactive_generation(
    config_path, checkpoint_path, tokenizer_path, max_length, temperature, top_k, device
):
    """编排交互式文本生成任务。"""
    setup_logging()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, config, tokenizer.vocab_size)

    print("\n" + "=" * 50)
    print("Interactive Generation Mode (type 'quit' to exit)")
    print("=" * 50)
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == "quit":
            break
        print("\nGenerating...")
        output = generate_text(
            model, tokenizer, prompt, max_length, temperature, top_k, device
        )
        print("-" * 20 + " Output " + "-" * 20)
        print(output)
        print("-" * 48)
