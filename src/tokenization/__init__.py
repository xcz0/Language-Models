# data_pipeline/tokenization/__init__.py

from .base_tokenizer import BaseTokenizer
from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer

Tokenizer_REGISTRY = {
    "CharTokenizer": CharTokenizer,
    "BPETokenizer": BPETokenizer,
    # "rnn_nlm": RNN_NLM,
    # "transformer_lm": TransformerLM,
}
