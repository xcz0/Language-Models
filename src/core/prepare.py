# src/core/preparation.py

import logging
from pathlib import Path
from typing import Tuple
import torch
import yaml

from src.data_pipeline import create_data_processor
from src.tokenization import CharTokenizer
from src.utils.io import load_processed_data

logger = logging.getLogger(__name__)


def process_and_save_data(
    config: dict, force_reprocess: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """
    执行数据预处理的核心流程：创建处理器，处理数据并返回结果。
    """
    try:
        data_path = Path(config["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        params = {
            "data_path": str(data_path),
            "tokenizer_path": config["tokenizer_path"],
            "processed_data_path": config["processed_data_path"],
            "cleaner_type": config.get("cleaner_type", "poetry"),
            "content_field": config.get("content_field", "contents"),
        }
        context_window = config["training_params"]["context_window"]

        logger.info(f"Creating data processor with params: {params}")
        processor = create_data_processor(**params)

        logger.info("Processing data...")
        X, Y, tokenizer = processor.get_or_process_data(
            context_window=context_window, force_reprocess=force_reprocess
        )
        logger.info(
            f"Data processing complete. Tensor shape: {X.shape}, Vocab size: {tokenizer.vocab_size}"
        )
        return X, Y, tokenizer
    except KeyError as e:
        raise ValueError(f"Configuration is missing a required key: {e}")
    except Exception as e:
        logger.error(f"Failed during data processing: {e}")
        raise


def validate_processed_data(X: torch.Tensor, Y: torch.Tensor, tokenizer: CharTokenizer):
    """验证处理后的数据。"""
    logger.info("Validating processed data...")
    assert X.shape == Y.shape, f"Tensor shape mismatch: {X.shape} vs {Y.shape}"
    assert X.numel() > 0, "Dataset is empty."
    vocab_size = tokenizer.vocab_size
    assert X.max() < vocab_size and Y.max() < vocab_size, (
        "Token index out of vocab range."
    )
    assert X.min() >= 0 and Y.min() >= 0, "Negative token index found."
    logger.info("✅ Data validation successful!")


def load_and_validate_data(config: dict):
    """加载并验证已存在的处理后数据。"""
    processed_data_path = Path(config["processed_data_path"])
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")

    logger.info(f"Loading existing data from {processed_data_path}")
    data = load_processed_data(processed_data_path)

    if not all(key in data for key in ["X", "Y", "tokenizer"]):
        raise ValueError("Processed data file is invalid or corrupted.")

    validate_processed_data(data["X"], data["Y"], data["tokenizer"])
    logger.info("✅ Existing data validation successful!")
