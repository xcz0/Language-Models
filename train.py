# scripts/train.py

import yaml
import argparse
from pprint import pprint
import torch

from src.data_module import PoetryDataModule


def main(config_path: str):
    """主函数，用于测试数据模块"""

    # 1. 加载配置文件
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration:")
    pprint(config)
    print("-" * 30)

    # 2. 初始化 PoetryDataModule
    print("Initializing DataModule...")
    data_module = PoetryDataModule(
        data_path=config["data_path"],
        tokenizer_path=config["tokenizer_path"],
        batch_size=config["training_params"]["batch_size"],
        context_window=config["training_params"]["context_window"],
    )

    # 3. 运行 prepare_data 和 setup
    # PyTorch Lightning Trainer 会自动调用这些，但我们手动调用以进行测试
    print("Running prepare_data()...")
    data_module.prepare_data()

    print("\nRunning setup()...")
    data_module.setup()
    print("-" * 30)

    # 4. 获取一个训练批次并检查其形状
    print("Fetching one training batch to verify...")
    train_loader = data_module.train_dataloader()
    x_batch, y_batch = next(iter(train_loader))

    print(f"Batch fetched successfully!")
    print(f"Shape of X (inputs): {x_batch.shape}")
    print(f"Shape of Y (targets): {y_batch.shape}")

    # 5. 验证内容
    print("\nVerifying batch content...")
    context_window = config["training_params"]["context_window"]
    # 预期形状应为 [batch_size, context_window]
    expected_shape = (config["training_params"]["batch_size"], context_window)
    assert x_batch.shape == expected_shape, (
        f"X shape mismatch! Expected {expected_shape}"
    )
    assert y_batch.shape == expected_shape, (
        f"Y shape mismatch! Expected {expected_shape}"
    )

    # 检查 y 是否是 x 的移位版本
    assert torch.equal(x_batch[0, 1:], y_batch[0, :-1])

    print("\nDecoded example from batch:")
    print("Input (x): ", data_module.tokenizer.decode(x_batch[0].tolist()))
    print("Target (y):", data_module.tokenizer.decode(y_batch[0].tolist()))

    print("\n✅ Phase 1: Basic infrastructure setup complete and verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test DataModule setup for Language Model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
