# src/cli.py

import argparse
import sys

from scripts.prepare import run_prepare
from scripts.train import run_training
from scripts.generate import run_generation, run_interactive_generation
from scripts.evaluate import run_evaluation


def run_cli():
    """
    配置并运行项目的命令行接口。
    """
    parser = argparse.ArgumentParser(
        description="Language Model Zoo - A project to learn various language models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- 'prepare' 命令 ---
    parser_prepare = subparsers.add_parser(
        "prepare", help="Prepare and preprocess data for training."
    )
    parser_prepare.add_argument(
        "--config", type=str, required=True, help="Path to the data config file."
    )
    parser_prepare.add_argument(
        "--force", action="store_true", help="Force reprocessing of data."
    )
    parser_prepare.add_argument(
        "--validate-only", action="store_true", help="Only validate existing data."
    )
    parser_prepare.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output."
    )

    # --- 'train' 命令 ---
    parser_train = subparsers.add_parser("train", help="Train a new model.")
    parser_train.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    parser_train.add_argument(
        "--test-data",
        action="store_true",
        help="Run data pipeline verification and exit.",
    )

    # --- 'generate' 命令 ---
    parser_generate = subparsers.add_parser(
        "generate", help="Generate text from a trained model."
    )
    parser_generate.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint."
    )
    parser_generate.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    parser_generate.add_argument(
        "--prompt", type=str, default="", help="Initial prompt for text generation."
    )
    parser_generate.add_argument(
        "--max-length", type=int, default=100, help="Max length of generated text."
    )
    parser_generate.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser_generate.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling."
    )
    parser_generate.add_argument(
        "--num-samples", type=int, default=1, help="Number of samples to generate."
    )
    parser_generate.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use.",
    )
    parser_generate.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode."
    )

    # --- 'evaluate' 命令 (新增) ---
    parser_evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model on the validation/test set."
    )
    parser_evaluate.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file).",
    )
    parser_evaluate.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file used to train the model.",
    )
    parser_evaluate.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for evaluation.",
    )

    args = parser.parse_args()

    # 根据命令分发任务
    try:
        if args.command == "prepare":
            success = run_prepare(
                config_path=args.config,
                force_reprocess=args.force,
                validate_only=args.validate_only,
                verbose=args.verbose,
            )
            sys.exit(0 if success else 1)

        elif args.command == "train":
            run_training(config_path=args.config, test_data_only=args.test_data)

        elif args.command == "generate":
            # 从config中推断tokenizer路径，或允许覆盖
            # 这是一个可以改进的地方，目前简化处理
            tokenizer_path = "data/artifacts/tokenizer.pkl"

            if args.interactive:
                run_interactive_generation(
                    config_path=args.config,
                    checkpoint_path=args.checkpoint,
                    tokenizer_path=tokenizer_path,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=args.device,
                )
            else:
                run_generation(
                    config_path=args.config,
                    checkpoint_path=args.checkpoint,
                    tokenizer_path=tokenizer_path,
                    prompt=args.prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    num_samples=args.num_samples,
                    device=args.device,
                )
        elif args.command == "evaluate":
            run_evaluation(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                device=args.device,
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        # 可选：打印更详细的traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
