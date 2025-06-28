# src/utils/model_loader.py

import pickle
import logging
from pathlib import Path
import torch

from src.lit_language_model import LitLanguageModel
from src.model import MODEL_REGISTRY
from src.tokenization.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_path: str) -> BaseTokenizer:
    """从文件加载分词器。"""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    logger.info(f"Loading tokenizer from: {path}")
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str, config: dict, vocab_size: int
) -> LitLanguageModel:
    """从checkpoint文件加载完整的LightningModule。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    logger.info(f"Loading model from checkpoint: {path}")
    model_name = config["model_params"]["name"]
    ModelClass = MODEL_REGISTRY.get(model_name)
    if not ModelClass:
        raise ValueError(f"Unknown model type '{model_name}' in config.")

    base_model = ModelClass(vocab_size=vocab_size)

    lit_model = LitLanguageModel.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        learning_rate=config["training_params"]["learning_rate"],
        map_location="cpu",  # 确保在没有GPU的机器上也能加载
    )
    lit_model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return lit_model
