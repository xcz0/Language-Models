# main.py

import argparse
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中，以确保 'from src...' 可以正常工作
sys.path.append(str(Path(__file__).resolve().parent))

from src.trainer_cli import run_training


def main():
    """
    项目主入口点，使用 subparsers 分发命令。
    """
    parser = argparse.ArgumentParser(
        description="Language Model Zoo - A project to learn various language models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # 创建子命令解析器
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- 'train' 命令 ---
    parser_train = subparsers.add_parser(
        "train", help="Train a new model or resume training."
    )
    parser_train.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/base_config.yaml).",
    )
    parser_train.add_argument(
        "--test-data",
        action="store_true",  # 当出现这个参数时，其值为 True
        help="A flag to only run the data pipeline verification and then exit.",
    )

    # --- 'generate' 命令 (为未来预留) ---
    # parser_generate = subparsers.add_parser(
    #     "generate",
    #     help="Generate text from a trained model checkpoint."
    # )
    # parser_generate.add_argument(...) # 例如 --checkpoint_path, --start_text, --length

    # 解析参数
    args = parser.parse_args()

    # 根据命令分发任务
    if args.command == "train":
        run_training(config_path=args.config, test_data_only=args.test_data)
    # elif args.command == "generate":
    #     run_generation(...) # 调用生成文本的函数
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
