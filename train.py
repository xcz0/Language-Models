# src/train.py

import os
import argparse
import pickle
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from loguru import logger

from src.data import CharDataModule
from src.models import get_model_class, ModelConfig


def deep_update(source, overrides):
    """
    递归更新字典。
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source


def load_config(config_path: str) -> dict:
    """加载基础配置并与指定的配置文件合并。"""
    base_config_path = "configs/base_config.yaml"

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded base config from {base_config_path}")

    with open(config_path, "r") as f:
        override_config = yaml.safe_load(f)

    logger.info(f"Loaded and merging override config from {config_path}")

    config = deep_update(config, override_config)
    return config


def main(config: dict):
    """主训练函数"""

    # 1. 设置随机种子
    seed_everything(config["system"]["seed"], workers=True)

    # 2. 初始化 DataModule
    datamodule = CharDataModule(
        data_dir=config["data"]["data_dir"],
        input_file=config["data"]["input_file"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    datamodule.setup()

    # 3. 保存词汇表供 sample.py 使用
    work_dir = config["system"]["work_dir"]
    os.makedirs(work_dir, exist_ok=True)
    vocab_path = os.path.join(work_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({"chars": datamodule.chars}, f)
    logger.info(f"Vocabulary saved to {vocab_path}")

    # 4. 创建模型配置
    model_params = config["model"].copy()
    model_type = model_params.pop("type")  # 移除 type 参数，单独保存
    model_config = ModelConfig(
        vocab_size=datamodule.vocab_size,
        block_size=datamodule.block_size,
        **model_params,  # 传入除了 type 之外的模型参数
    )

    # 5. 创建优化器配置
    from src.models.base import AdamConfig

    optim_config = AdamConfig(
        learning_rate=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # 6. 实例化模型
    model_class = get_model_class(model_type)
    model = model_class(
        config=model_config,
        optim_config=optim_config,
    )
    logger.info(
        f"Initialized model '{model_type}' with {sum(p.numel() for p in model.parameters()):,} parameters."
    )

    # 7. 配置 Callbacks 和 Logger
    tb_logger = TensorBoardLogger(save_dir=work_dir, name=None, version="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{step}-{val_loss:.2f}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # 8. 初始化 Trainer
    trainer = Trainer(
        max_steps=config["training"]["max_steps"],
        val_check_interval=config["training"]["eval_interval"],
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        **config["training"]["trainer_args"],  # 传入 trainer 特定参数
    )

    # 9. 启动训练
    logger.info(f"Starting training for {config['training']['max_steps']} steps...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a character-level language model using YAML configs."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the experiment config file (e.g., 'configs/bigram.yaml')",
    )
    args = parser.parse_args()

    # 加载和合并配置
    config_data = load_config(args.config)

    main(config_data)
