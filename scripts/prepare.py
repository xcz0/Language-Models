# scripts/prepare.py

import logging
import yaml
from src.core.preparation import (
    process_and_save_data,
    validate_processed_data,
    load_and_validate_data,
)
from src.utils.config import load_config
from src.utils.log import setup_logging

logger = logging.getLogger(__name__)


def run_prepare(
    config_path: str, force_reprocess: bool, validate_only: bool, verbose: bool
) -> bool:
    """编排数据预处理流程。"""
    setup_logging("DEBUG" if verbose else "INFO")

    try:
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        if validate_only:
            logger.info("Running in validation-only mode.")
            load_and_validate_data(config)
        else:
            logger.info("Running full data preparation process.")
            X, Y, tokenizer = process_and_save_data(config, force_reprocess)
            validate_processed_data(X, Y, tokenizer)

        logger.info("✅ Prepare script finished successfully.")
        return True

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=verbose)
        return False
