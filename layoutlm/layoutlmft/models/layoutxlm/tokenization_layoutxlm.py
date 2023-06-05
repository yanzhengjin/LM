# coding=utf-8

from transformers import XLMRobertaTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "layoutxlm-base": "https://huggingface.co/layoutxlm-base/resolve/main/sentencepiece.bpe.model",
        "layoutxlm-large": "https://huggingface.co/layoutxlm-large/resolve/main/sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "layoutxlm-base": 512,
    "layoutxlm-large": 512,
}


class LayoutXLMTokenizer(XLMRobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_max_length=512, **kwargs):
        super().__init__(model_max_length=model_max_length, **kwargs)
# 这段代码定义了一个 LayoutXLMTokenizer 类，继承自 XLMRobertaTokenizer 类。
# 该类用于实例化一个 tokenizer 对象，将文本数据转化为模型能够接受的数字输入。
# 其中：vocab_files_names 定义了该 tokenizer 使用的词汇表文件名；
# pretrained_vocab_files_map 定义了预训练模型的词汇表文件名和对应的 URL 地址；
# max_model_input_sizes 定义了该 tokenizer 对应的预训练模型能够接受的最大输入长度；
# model_input_names 定义了模型接收的输入名称；
# __init__ 方法对 model_max_length 进行了设置，可以控制 tokenizer 能够接受的最大文本长度。