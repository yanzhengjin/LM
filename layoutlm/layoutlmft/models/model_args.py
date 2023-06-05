from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # 这是一个用于传递模型、配置和tokenizer相关参数的类，包含以下字段:
    # model_name_or_path: 一个字符串类型的参数，表示预训练模型的路径或模型标识符。
    # config_name: 一个可选的字符串类型的参数，表示预训练配置的名称或路径，如果与model_name不同，则需要设置。
    # tokenizer_name: 一个可选的字符串类型的参数，表示预训练tokenizer的名称或路径，如果与model_name不同，则需要设置。
    # cache_dir: 一个可选的字符串类型的参数，表示缓存目录，用于存储从huggingface.co下载的预训练模型。
    # model_revision: 一个字符串类型的参数，表示要使用的特定模型版本(可以是分支名称、标记名称或提交id)。
    # use_auth_token: 一个布尔类型的参数，表示是否使用在运行transformers-cli login时生成的令牌(必须使用此脚本与私有模型)。