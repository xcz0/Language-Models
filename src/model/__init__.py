# src/models/__init__.py

from .bigram import BigramModel
# 当你添加新模型时，在这里导入它们
# from .rnn_nlm import RNN_NLM
# from .transformer_lm import TransformerLM

# 创建一个模型注册表，方便通过字符串名称获取模型类
MODEL_REGISTRY = {
    "bigram": BigramModel,
    # "rnn_nlm": RNN_NLM,
    # "transformer_lm": TransformerLM,
}
