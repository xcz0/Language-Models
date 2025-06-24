# main.py

import json
from pathlib import Path
from data_pipeline import create_data_processor


# --- 准备模拟数据 ---
def setup_dummy_data():
    """创建模拟数据文件以便脚本可以运行。"""
    data_dir = Path("./data/temp_data")
    data_dir.mkdir(exist_ok=True)

    poetry_data_path = data_dir / "poetry.json"
    general_data_path = data_dir / "general_text.json"

    poetry_data = [
        {"contents": "静夜思  床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        {"contents": "春晓  春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"},
    ]
    general_data = [
        {"text": "这是一段测试文本，包含<html>标签</html>和   多余的空格。"},
        {"text": "另一段文本，用于测试\n\n段落保留功能。"},
    ]

    with open(poetry_data_path, "w", encoding="utf-8") as f:
        json.dump(poetry_data, f, ensure_ascii=False, indent=2)
    with open(general_data_path, "w", encoding="utf-8") as f:
        json.dump(general_data, f, ensure_ascii=False, indent=2)

    print(f"模拟数据已创建在 {data_dir.resolve()} 目录下。")
    return data_dir


def run_poetry_processing(data_dir: Path):
    """示例：处理诗歌数据。"""
    print("\n" + "=" * 20 + " 处理诗歌数据 " + "=" * 20)
    processor = create_data_processor(
        data_path=str(data_dir / "poetry.json"),
        tokenizer_path=str(data_dir / "poetry_tokenizer.pkl"),
        processed_data_path=str(data_dir / "poetry_processed.pkl"),
        cleaner_type="poetry",
        content_field="contents",
    )

    X, Y, tokenizer = processor.get_or_process_data(
        context_window=10, force_reprocess=True
    )

    print(f"\n处理完成！")
    print(f"输入张量形状: {X.shape}")
    print(f"目标张量形状: {Y.shape}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(
        f"示例解码: '{tokenizer.decode(X[0].tolist())}' -> '{tokenizer.decode(Y[0].tolist())}'"
    )


def run_custom_processing(data_dir: Path):
    """示例：使用自定义清洗器处理诗歌数据。"""
    print("\n" + "=" * 20 + " 使用自定义清洗器 " + "=" * 20)

    custom_patterns = [{"pattern": r"静夜思|春晓", "replacement": "[诗名]"}]
    custom_replacements = {"。": "！"}

    processor = create_data_processor(
        data_path=str(data_dir / "poetry.json"),
        tokenizer_path=str(data_dir / "custom_tokenizer.pkl"),
        processed_data_path=str(data_dir / "custom_processed.pkl"),
        cleaner_type="custom",
        content_field="contents",
        custom_patterns=custom_patterns,
        custom_replacements=custom_replacements,
    )

    X, Y, tokenizer = processor.get_or_process_data(
        context_window=10, force_reprocess=True
    )

    print(f"\n处理完成！")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    # 检查清洗效果
    raw_text = "[诗名] 床前明月光，疑是地上霜！举头望明月，低头思故乡！"
    print(
        f"查看自定义清洗效果，'。' 是否被替换为 '！': {raw_text in tokenizer.decode(X.flatten().tolist())}"
    )


if __name__ == "__main__":
    temp_data_dir = setup_dummy_data()
    run_poetry_processing(temp_data_dir)
    run_custom_processing(temp_data_dir)
