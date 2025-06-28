# scripts/evaluate.py

import logging
from pathlib import Path
import torch
import pytorch_lightning as pl

from src.data_module import DataModule
from src.utils.config import load_config
from src.utils.log import setup_logging
from src.utils.model_loader import load_tokenizer, load_model_from_checkpoint

logger = logging.getLogger(__name__)


def run_evaluation(config_path: str, checkpoint_path: str, device: str):
    """
    编排模型评估流程。

    此函数会加载一个训练好的模型 checkpoint，准备测试/验证数据集，
    并使用 PyTorch Lightning Trainer 在该数据集上运行评估，
    最后打印出评估指标（如损失和困惑度）。

    Args:
        config_path: 用于训练的配置文件路径。
        checkpoint_path: 模型的 .ckpt 文件路径。
        device: 使用的设备 ('auto', 'cpu', 'cuda')。
    """
    setup_logging()

    try:
        # 1. 加载配置
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)

        # 2. 确定设备
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 3. 加载分词器（从数据模块中获取）
        # 我们需要数据模块来加载数据，它内部会加载分词器
        logger.info("Setting up DataModule to access tokenizer and test data...")
        data_module = DataModule(
            processed_data_path=Path(config["processed_data_path"]),
            batch_size=config["training_params"]["batch_size"],
        )
        data_module.prepare_data()
        data_module.setup("test")  # 'test' 阶段会准备 val_dataloader

        if data_module.tokenizer is None:
            raise RuntimeError("Tokenizer could not be loaded via DataModule.")

        logger.info(
            f"Tokenizer loaded with vocab size: {data_module.tokenizer.vocab_size}"
        )

        # 4. 从 Checkpoint 加载模型
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        lit_model = load_model_from_checkpoint(
            checkpoint_path, config, data_module.tokenizer.vocab_size
        )

        # 5. 初始化 Trainer
        # 对于评估，我们不需要复杂的设置，只需要指定设备
        trainer = pl.Trainer(
            accelerator="gpu" if device == "cuda" else "cpu",
            devices=1,
            logger=False,  # 我们只想在控制台看到结果，不创建新的日志文件
        )

        # 6. 运行评估
        # trainer.test 会调用 LitLanguageModel 中的 test_step
        # 如果没有 test_dataloader，它会默认使用 val_dataloader
        logger.info("🚀 Starting model evaluation... 🚀")

        # 为了清晰，我们显式地使用验证集进行评估
        # 如果你有专门的测试集，请确保 DataModule.setup() 和 test_dataloader() 已正确实现
        results = trainer.test(
            model=lit_model, dataloaders=data_module.val_dataloader()
        )

        logger.info("✅ Evaluation completed successfully!")

        # 结果是一个包含字典的列表，我们打印第一个字典
        if results:
            print("\n" + "=" * 20 + " Evaluation Results " + "=" * 20)
            for key, value in results[0].items():
                print(f"{key:<20}: {value:.4f}")
            print("=" * 62)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Please check your paths.")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during evaluation: {e}", exc_info=True
        )
        raise
