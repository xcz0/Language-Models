#!/usr/bin/env python3
# scripts/prepare.py

"""
数据预处理脚本
负责数据的加载、清洗、分词和序列化处理，为模型训练做准备。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple
import yaml
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.processing import create_data_processor
from src.data_pipeline.tokenization import CharTokenizer


def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("prepare.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def prepare_data(
    config: dict, force_reprocess: bool = False, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """
    执行数据预处理流程

    Args:
        config: 配置字典
        force_reprocess: 是否强制重新处理
        verbose: 是否详细输出

    Returns:
        (输入张量, 目标张量, 分词器)
    """
    logger = logging.getLogger(__name__)

    # 从配置中提取参数
    data_path = config["data_path"]
    tokenizer_path = config["tokenizer_path"]
    processed_data_path = config["processed_data_path"]
    cleaner_type = config.get("cleaner_type", "poetry")
    content_field = config.get("content_field", "contents")
    context_window = config["training_params"]["context_window"]

    logger.info("开始数据预处理...")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"分词器路径: {tokenizer_path}")
    logger.info(f"处理后数据路径: {processed_data_path}")
    logger.info(f"清洗器类型: {cleaner_type}")
    logger.info(f"上下文窗口: {context_window}")

    # 创建数据处理器
    processor = create_data_processor(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        processed_data_path=processed_data_path,
        cleaner_type=cleaner_type,
        content_field=content_field,
    )

    # 执行数据处理
    X, Y, tokenizer = processor.get_or_process_data(
        context_window=context_window, force_reprocess=force_reprocess
    )

    logger.info("数据处理完成!")
    logger.info(f"输入张量形状: {X.shape}")
    logger.info(f"目标张量形状: {Y.shape}")
    logger.info(f"词汇表大小: {tokenizer.vocab_size}")

    return X, Y, tokenizer


def validate_data(X: torch.Tensor, Y: torch.Tensor, tokenizer: CharTokenizer):
    """验证处理后的数据"""
    logger = logging.getLogger(__name__)

    logger.info("验证数据质量...")

    # 基本形状检查
    assert X.shape == Y.shape, f"输入和目标形状不匹配: {X.shape} vs {Y.shape}"
    assert X.shape[0] > 0, "数据集为空"
    assert X.shape[1] > 0, "序列长度为0"

    # 检查数据范围
    vocab_size = tokenizer.vocab_size
    assert X.min() >= 0, f"输入包含负值: {X.min()}"
    assert X.max() < vocab_size, f"输入超出词汇表范围: {X.max()} >= {vocab_size}"
    assert Y.min() >= 0, f"目标包含负值: {Y.min()}"
    assert Y.max() < vocab_size, f"目标超出词汇表范围: {Y.max()} >= {vocab_size}"

    # 检查序列偏移关系 (Y应该是X的下一个字符)
    sample_idx = 0
    logger.debug(f"样本 {sample_idx} 输入序列: {X[sample_idx][:10].tolist()}")
    logger.debug(f"样本 {sample_idx} 目标序列: {Y[sample_idx][:10].tolist()}")

    # 解码一个样本进行检查
    input_text = tokenizer.decode(X[sample_idx][:20].tolist())
    target_text = tokenizer.decode(Y[sample_idx][:20].tolist())
    logger.debug(f"输入文本样本: '{input_text}'")
    logger.debug(f"目标文本样本: '{target_text}'")

    logger.info("数据验证通过!")


def run_prepare(
    config_path: str,
    force_reprocess: bool = False,
    validate_only: bool = False,
    verbose: bool = False,
):
    """
    运行数据预处理流程

    Args:
        config_path: 配置文件路径
        force_reprocess: 是否强制重新处理
        validate_only: 是否仅进行验证
        verbose: 是否详细输出
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # 加载配置
        config = load_config(config_path)

        if validate_only:
            logger.info("仅进行数据验证...")
            # 尝试加载已存在的数据进行验证
            from src.data_pipeline.utils.io import load_processed_data

            processed_data_path = Path(config["processed_data_path"])
            if not processed_data_path.exists():
                logger.error("找不到已处理的数据文件，请先运行数据预处理")
                return False

            data = load_processed_data(processed_data_path)
            validate_data(data["X"], data["Y"], data["tokenizer"])
        else:
            # 执行完整的数据预处理流程
            X, Y, tokenizer = prepare_data(
                config=config, force_reprocess=force_reprocess, verbose=verbose
            )

            # 验证数据
            validate_data(X, Y, tokenizer)

        logger.info("数据预处理流程完成!")
        return True

    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        if verbose:
            import traceback

            logger.error(traceback.format_exc())
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="数据预处理脚本", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (例如: configs/base_config.yaml)",
    )

    parser.add_argument(
        "--force", action="store_true", help="强制重新处理数据，忽略已存在的文件"
    )

    parser.add_argument(
        "--validate-only", action="store_true", help="仅验证已存在的数据，不进行预处理"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    success = run_prepare(
        config_path=args.config,
        force_reprocess=args.force,
        validate_only=args.validate_only,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
