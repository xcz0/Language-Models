# src/core/training.py

import logging
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_module import DataModule
from src.lit_language_model import LitLanguageModel
from src.model import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def setup_training_components(config: dict) -> tuple:
    """
    初始化并返回 DataModule, LitLanguageModel, 和 Trainer。
    """
    # 1. 设置 DataModule
    data_path_str = config["processed_data_path"]
    data_module = DataModule(
        processed_data_path=Path(data_path_str),
        batch_size=config["training_params"]["batch_size"],
    )
    # 调用 prepare_data 和 setup 是必须的，以加载数据和分词器
    data_module.prepare_data()
    data_module.setup()

    if data_module.tokenizer is None:
        raise RuntimeError(
            f"Tokenizer could not be loaded from {data_path_str}. "
            f"Please run the 'prepare' command first."
        )
    logger.info(f"DataModule initialized from {data_module.processed_data_path}")

    # 2. 设置模型 (LitLanguageModel)
    model_name = config["model_params"]["name"]
    ModelClass = MODEL_REGISTRY.get(model_name)
    if not ModelClass:
        raise ValueError(f"Model '{model_name}' not found.")

    model = ModelClass(vocab_size=data_module.tokenizer.vocab_size)
    lit_model = LitLanguageModel(
        model, learning_rate=config["training_params"]["learning_rate"]
    )
    logger.info(
        f"Model '{model_name}' initialized with vocab size {data_module.tokenizer.vocab_size}"
    )

    # 3. 设置 Trainer 和 Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(**config["trainer_params"], callbacks=[checkpoint_callback])

    return data_module, lit_model, trainer
