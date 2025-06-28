# scripts/train.py

import yaml
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_module import DataModule
from src.lit_language_model import LitLanguageModel
from src.model import MODEL_REGISTRY


def run_training(config_path: str, test_data_only: bool = False):
    """
    'train' 命令的核心逻辑。
    加载配置, 初始化所有模块, 并启动训练。

    注意：此脚本现在要求数据已经预处理完成。如果处理后的数据文件不存在，
    请先运行数据预处理脚本：
        python scripts/prepare.py --config configs/your_config.yaml

    Args:
        config_path: 配置文件路径
        test_data_only: 如果为 True，只运行数据管道验证然后退出
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
    # 注意：新的 DataModule 只加载已处理的数据，不执行预处理
    # 确保 processed_data_path 指向的文件已存在
    try:
        data_module = DataModule(
            processed_data_path=config.get(
                "processed_data_path", "data/processed_data.pkl"
            ),
            batch_size=config["training_params"]["batch_size"],
            rate=config.get("train_val_split", 0.9),  # 训练/验证集分割比例
        )
        # 检查数据文件并加载数据
        data_module.prepare_data()
        data_module.setup()
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 解决方案：请先运行数据预处理脚本生成处理后的数据文件:")
        print(f"   python scripts/prepare.py --config {config_path}")
        return

    # 如果只是测试数据管道，在这里退出
    if test_data_only:
        print("\n✅ Data pipeline verification completed successfully!")
        if data_module.tokenizer:
            print(f"Tokenizer vocab size: {data_module.tokenizer.vocab_size}")
        else:
            print("Tokenizer not yet initialized")
        print("Exiting as requested (--test-data flag was used).")
        return

    # 3. 初始化模型
    print("\nInitializing Model...")
    model_name = config["model_params"]["name"]
    ModelClass = MODEL_REGISTRY.get(model_name)
    if not ModelClass:
        raise ValueError(
            f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # 从 data_module 获取 vocab_size
    if data_module.tokenizer is None:
        raise RuntimeError(
            "Tokenizer not initialized. Make sure to call setup() first."
        )
    vocab_size = data_module.tokenizer.vocab_size
    model = ModelClass(vocab_size=vocab_size)
    print(f"Model '{model_name}' with vocab_size={vocab_size} initialized.")

    # 4. 初始化 LightningModule
    print("\nInitializing LightningModule...")
    lit_model = LitLanguageModel(
        model, learning_rate=config["training_params"]["learning_rate"]
    )

    # 5. 初始化 Trainer
    print("\nInitializing Trainer...")
    # 设置模型检查点回调，只保存效果最好的模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(**config["trainer_params"], callbacks=[checkpoint_callback])

    # 6. 启动训练！
    print("\n🚀 Starting training! 🚀")
    trainer.fit(model=lit_model, datamodule=data_module)
    print("\n✅ Training finished!")
