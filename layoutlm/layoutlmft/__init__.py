from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)


CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig)])
MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM")])
TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)),
    ]
)
    # 它做了两个主要的事情：
    # 更新 TOKENIZER_MAPPING 字典的内容。将两个元组 (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)) 和 (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)) 添加到 TOKENIZER_MAPPING 字典中。
    # 在这个字典中，每个键都是一个 Config 类的子类，每个值都是一个包含两个元素的元组。
    # 这两个元素都是 Tokenizer 类的子类，分别是普通的 Tokenizer 和快速的 Tokenizer。
    # 这段代码的目的是更新 TOKENIZER_MAPPING 字典，以便在使用不同的配置类时，能够正确地选择使用适当的 Tokenizer。
    # 具体来说，它添加了两个新的键值对，其中键是 LayoutLMv2Config 和 LayoutXLMConfig，而值是两个元素的元组，其中包含相应的 Tokenizer 子类。
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LayoutLMv2Config, LayoutLMv2ForTokenClassification), (LayoutXLMConfig, LayoutXLMForTokenClassification)]
)
    # 这段代码是 Python 代码，它做了两个主要的事情：
    # 更新 SLOW_TO_FAST_CONVERTERS 字典的内容。
    # 更新 MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 字典的内容。
    # 首先，SLOW_TO_FAST_CONVERTERS 字典是用于将“慢速”Tokenizer转换为“快速”Tokenizer的字典，其中键是“慢速”Tokenizer的名称，值是实现转换的类。
    # 在这个代码段中，它将键 LayoutLMv2Tokenizer 和 LayoutXLMTokenizer 分别映射到 BertConverter 和 XLMRobertaConverter 类。
    # 其次，MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 字典是一个字典，用于根据配置类选择适当的模型。
    # 这个字典的键是配置类，值是相应的模型。
    # 在这个代码段中，它添加了两个新的键值对，其中键是 LayoutLMv2Config 和 LayoutXLMConfig，值是相应的模型类 LayoutLMv2ForTokenClassification 和 LayoutXLMForTokenClassification。
    # 这样，当使用 LayoutLMv2Config 或 LayoutXLMConfig 配置类时，就可以选择适当的模型进行标记分类任务。
MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LayoutLMv2Config, LayoutLMv2ForRelationExtraction), (LayoutXLMConfig, LayoutXLMForRelationExtraction)]
)
     # MODEL_FOR_RELATION_EXTRACTION_MAPPING，该字典用于将配置类映射到相应的关系抽取模型。
     # 具体来说，字典中包含两个键值对，分别是 (LayoutLMv2Config, LayoutLMv2ForRelationExtraction) 和 (LayoutXLMConfig, LayoutXLMForRelationExtraction)。
    # 这表示当使用 LayoutLMv2Config 或 LayoutXLMConfig 配置类时，可以选择相应的模型类 LayoutLMv2ForRelationExtraction 或 LayoutXLMForRelationExtraction 进行关系抽取任务。
    # 此外，由于该字典是有序的，因此在使用时可以确保按照添加到字典中的顺序选择模型类。
AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)
    # 这段代码定义了两个类工厂函数 auto_class_factory，用于根据给定的字典和类名创建新的自动模型类。
    # 具体来说，第一个类工厂函数 AutoModelForTokenClassification 通过调用 auto_class_factory 函数创建一个名为 AutoModelForTokenClassification 的新类。
    # 这个新类的作用是自动选择适当的模型类，以便执行标记分类任务。
    # 为此，它将 MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 字典传递给 auto_class_factory 函数，并将其命名为 head_doc。
    # 这个 head_doc 参数表示新类的文档字符串中包含的模型头部类型的描述。
AutoModelForRelationExtraction = auto_class_factory(
    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
)
    # 同样，第二个类工厂函数 AutoModelForRelationExtraction 通过调用 auto_class_factory 函数创建一个名为 AutoModelForRelationExtraction 的新类。
    # 这个新类的作用是自动选择适当的模型类，以便执行关系抽取任务。
    # 为此，它将 MODEL_FOR_RELATION_EXTRACTION_MAPPING 字典传递给 auto_class_factory 函数，并将其命名为 head_doc。
    # 这个 head_doc 参数表示新类的文档字符串中包含的模型头部类型的描述。