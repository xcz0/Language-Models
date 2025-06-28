# scripts/train.py

import logging
import yaml
from src.core.training import setup_training_components
from src.utils.config import load_config
from src.utils.log import setup_logging

logger = logging.getLogger(__name__)


def run_training(config_path: str, test_data_only: bool):
    """编排模型训练流程。"""
    setup_logging()

    try:
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        logger.info("Setting up training components...")
        data_module, lit_model, trainer = setup_training_components(config)

        if test_data_only:
            logger.info("Data pipeline verification successful. Exiting as requested.")
            return

        logger.info("🚀 Starting training! 🚀")
        trainer.fit(model=lit_model, datamodule=data_module)
        logger.info("✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
