# data_pipeline/tokenization/examples.py

"""
分词器使用示例，展示基类重构后的统一接口
"""

import logging
from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)


def demo_char_tokenizer():
    """演示字符级分词器"""
    print("=== 字符级分词器演示 ===")

    # 创建分词器
    tokenizer = CharTokenizer()

    # 训练
    sample_text = "Hello, 世界! This is a test. 这是一个测试。"
    tokenizer.fit(sample_text)

    # 编码和解码
    test_text = "Hello 世界"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"分词器状态: {tokenizer}")

    # 获取文本统计
    stats = tokenizer.get_text_stats(test_text)
    print(f"文本统计: {stats}")

    # 保存和加载
    tokenizer.save("/tmp/char_tokenizer.pkl")

    # 创建新分词器并加载
    new_tokenizer = CharTokenizer()
    new_tokenizer.load("/tmp/char_tokenizer.pkl")

    # 验证加载是否成功
    test_encoded = new_tokenizer.encode(test_text)
    print(f"加载后编码结果: {test_encoded}")
    print(f"编码结果一致: {encoded == test_encoded}")


def demo_bpe_tokenizer():
    """演示BPE分词器"""
    print("\n=== BPE分词器演示 ===")

    # 创建分词器
    tokenizer = BPETokenizer()

    # 训练
    sample_text = "Hello world! This is a simple test for BPE tokenizer. " * 100
    tokenizer.fit(sample_text, vocab_size=500)

    # 编码和解码
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"分词器状态: {tokenizer}")

    # 获取文本统计
    stats = tokenizer.get_text_stats(test_text)
    print(f"文本统计: {stats}")

    # 保存和加载
    tokenizer.save("/tmp/bpe_tokenizer.pkl")

    # 创建新分词器并加载
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("/tmp/bpe_tokenizer.pkl")

    # 验证加载是否成功
    test_encoded = new_tokenizer.encode(test_text)
    print(f"加载后编码结果: {test_encoded}")
    print(f"编码结果一致: {encoded == test_encoded}")


def demo_polymorphic_usage():
    """演示多态使用"""
    print("\n=== 多态使用演示 ===")

    def test_tokenizer(tokenizer, name, text, **fit_kwargs):
        """测试分词器的通用函数"""
        print(f"\n--- 测试 {name} ---")

        if not tokenizer.is_fitted:
            tokenizer.fit(text, **fit_kwargs)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        stats = tokenizer.get_text_stats(text)

        print(f"词汇表大小: {tokenizer.vocab_size}")
        print(f"编码长度: {len(encoded)}")
        print(f"压缩比: {stats.get('压缩比', 'N/A')}")
        print(f"重构是否成功: {decoded == text}")

    # 测试文本
    test_text = "Hello world! 你好世界！"

    # 创建不同类型的分词器
    tokenizers = [
        (CharTokenizer(), "字符级分词器", {}),
        (BPETokenizer(), "BPE分词器", {"vocab_size": 300}),
    ]

    # 多态测试
    for tokenizer, name, fit_kwargs in tokenizers:
        test_tokenizer(tokenizer, name, test_text, **fit_kwargs)


if __name__ == "__main__":
    demo_char_tokenizer()
    demo_bpe_tokenizer()
    demo_polymorphic_usage()
    print("\n🎉 所有演示完成！基类重构成功！")
