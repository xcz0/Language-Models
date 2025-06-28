# main.py

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中
sys.path.append(str(Path(__file__).resolve().parent))

from src.cli import run_cli

if __name__ == "__main__":
    run_cli()
