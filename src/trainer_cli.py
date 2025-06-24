# src/trainer_cli.py

import yaml
from pprint import pprint
import torch

from src.data_module import PoetryDataModule
# 在下一阶段，我们将取消下面这些行的注释
# import pytorch_lightning as pl
# from src.lit_language_model import LitLanguageModel
# from src.models.bigram import BigramModel # 举例


def run_training(config_path: str, test_data_only: bool = False):
    """
    'train' 命令的核心逻辑。
    加载配置, 初始化数据模块, 并根据标志决定是仅测试数据流还是启动完整训练。
    """
    # 1. 加载配置文件
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Configuration:")
    pprint(config)
    print("-" * 30)

    # 2. 初始化 DataModule
    print("Initializing DataModule...")
    data_module = PoetryDataModule(
        data_path=config["data_path"],
        tokenizer_path=config["tokenizer_path"],
        batch_size=config["training_params"]["batch_size"],
        context_window=config["training_params"]["context_window"],
    )

    # 3. 如果只是测试数据流，则执行验证并退出
    if test_data_only:
        print("Running in --test-data mode. Verifying data pipeline...")
        _verify_data_pipeline(data_module, config)
        return

    # --- 完整的训练流程 (将在第二阶段完成) ---
    print("\nSetting up for a full training run...")

    # 这里是为下一阶段准备的占位符
    print("TODO: Instantiate Model (e.g., BigramModel)")
    print("TODO: Instantiate LightningModule (LitLanguageModel)")
    print("TODO: Instantiate Lightning Trainer")
    print("TODO: Call trainer.fit()")

    print("\n✅ Full training script setup is ready for Phase 2.")


def _verify_data_pipeline(data_module: PoetryDataModule, config: dict):
    """
    一个私有辅助函数，封装了数据验证的逻辑。
    """
    data_module.prepare_data()
    data_module.setup()
    print("-" * 30)

    print("Fetching one training batch to verify...")
    try:
        train_loader = data_module.train_dataloader()
        x_batch, y_batch = next(iter(train_loader))
    except Exception as e:
        print(f"Error fetching batch: {e}")
        print("Please check your DataLoader and Dataset implementation.")
        return

    print(f"Batch fetched successfully!")
    print(f"Shape of X (inputs): {x_batch.shape}")
    print(f"Shape of Y (targets): {y_batch.shape}")

    context_window = config["training_params"]["context_window"]
    batch_size = config["training_params"]["batch_size"]

    # 动态调整预期的批次大小，以防最后一个批次不完整
    effective_batch_size = x_batch.shape[0]
    expected_shape = (effective_batch_size, context_window)

    assert x_batch.shape == expected_shape, (
        f"X shape mismatch! Expected {expected_shape}, got {x_batch.shape}"
    )
    assert y_batch.shape == expected_shape, (
        f"Y shape mismatch! Expected {expected_shape}, got {y_batch.shape}"
    )
    assert torch.equal(x_batch[0, 1:], y_batch[0, :-1])

    print("\nDecoded example from batch:")
    print("Input (x): ", data_module.tokenizer.decode(x_batch[0].tolist()))
    print("Target (y):", data_module.tokenizer.decode(y_batch[0].tolist()))

    print("\n✅ Data pipeline verification successful!")
