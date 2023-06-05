from dataclasses import dataclass
from typing import Optional, Union

import torch

from detectron2.structures import ImageList
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """
        # 这是一个数据处理函数的参数说明，用于为机器学习模型训练准备输入数据和标签。具体来说，该数据处理函数可以动态地填充训练中接收到的输入和标签，基于指定的参数设置。其中，参数的具体含义如下：
        # tokenizer：用于编码数据的分词器。
        # padding：选择一种策略来填充返回的序列（根据模型的填充方向和填充索引），包括“True”或“longest”（填充到批次中最长的序列），“max_length”（填充到指定的最大长度或模型可接受的最大输入长度），以及“False”或“do_not_pad”（不填充，可以输出长度不同的序列批次）。
        # max_length：返回列表的最大长度，同时也可以指定填充长度（参见上文）。
        # pad_to_multiple_of：如果设置，将填充序列到提供的值的倍数。这对于启用
        # NVIDIA硬件上的张量核心（Tensor Cores），并且具有计算能力 >= 7.5（Volta）尤其有用。
        # label_pad_token_id：填充标签时要使用的ID（-100将自动被PyTorch损失函数忽略）。

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
        # 是一个参数列表，其中包含用于初始化一个名为"tokenizer"的预训练分词器对象的参数。
        # 在这里，"PreTrainedTokenizerBase"是一个抽象类，可能是一个由预训练模型定义的特定分词器类的基类。因此，这个参数指定要使用哪个分词器。
        # "padding"是一个控制是否对输入进行填充的参数。可以是布尔值(True或False)，用于指示是否填充。还可以是字符串或PaddingStrategy枚举，用于指定填充策略。
        # "max_length"是一个可选参数，用于指定最大输入序列长度。如果输入序列超过该长度，则会进行截断或填充。
        # "pad_to_multiple_of"也是一个可选参数，用于指定填充长度应该是多少的倍数。
        # "label_pad_token_id"是一个整数，用于指定标签填充的tokenid。
        # 在某些模型中，可能需要将标签与填充token区分开来，因此需要将其指定为不同的值。默认情况下，此参数为 - 100。

    def __call__(self, features):
           #它的作用是从输入的特征中提取标签和图像边界框信息。
        label_name = "label" if "label" in features[0].keys() else "labels"  #检查特征列表中是否存在"label"键。如果存在，将label_name设置为"label"，否则设置为"labels"。
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
           #从特征列表中提取标签信息。如果存在标签，将它们存储在一个标签列表中，否则将标签列表设置为None。

        has_image_input = "image" in features[0]   #检查特征列表中是否存在"image"键。如果存在，将has_image_input设置为True，否则设置为False。
        has_bbox_input = "bbox" in features[0]     #特征列表中是否存在"bbox"键。如果存在，将has_bbox_input设置为True，否则设置为False。
        if has_image_input:
            image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            for feature in features:
                del feature["image"]
               # 如果has_image_input为真，说明输入中包含图像。则通过ImageList.from_tensors将所有图像数据转换成一个ImageList对象，以便于进行后续的图像处理。
               # 同时，在features中删除图像数据，以便于后续处理不包含图像数据。
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
               # 将剩下的输入特征使用tokenizer进行处理，将其填充到指定的max_length，并转换成PyTorch张量形式。
               # 如果输入中包含标签labels，则不能将特征数据转换成张量，因为标签数据和特征数据长度不同。最终返回处理后的张量形式的输入特征数据，供模型使用。

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]  #其中使用了PyTorch和Hugging Face Transformers库的函数。这段代码的目的是获取输入序列的长度，并确定padding的方向。
        padding_side = self.tokenizer.padding_side
           # 代码中第一行使用了PyTorch库的tensor函数，将batch["input_ids"]转换为一个PyTorch的张量，并使用shape函数获取了该张量的形状。
           # 这里，由于batch["input_ids"]是一个二维张量，因此通过索引1来获取其第二个维度，也就是输入序列的长度。这个长度信息通常用于后续的数据处理或模型训练中。
           # 第二行代码使用了Hugging Face Transformers库中Tokenizer类的padding_side属性，来确定padding的方向。
           # 当调用Tokenizer类的encode_plus函数时，如果输入的文本长度不足最大长度，会自动在文本的左边或右边填充一些特殊的token，以使其达到最大长度。padding_side属性用于指定padding的位置，
           # 可以设置为"left"或"right"，分别表示在文本左侧或右侧进行padding。
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
               # 用于对输入的标签序列进行padding，并且如果输入的数据中还包含bounding box信息，则对bounding box序列也进行padding。
               # 如果padding_side为"right"，则说明padding是在标签序列的右侧进行的，这时需要对每个标签序列label进行padding，将self.label_pad_token_id添加到label的末尾，直到序列的长度达到sequence_length。
               # 将这些padding后的标签序列存储在batch["labels"]中，并且如果输入数据中还包含bounding box信息，则对每个bounding box序列进行类似的padding。

               # 如果padding_side为"left"，则说明padding是在标签序列的左侧进行的，这时需要对每个标签序列label进行padding，将self.label_pad_token_id添加到label的开头，直到序列的长度达到sequence_length。
               # 将这些padding后的标签序列存储在batch["labels"]中，并且如果输入数据中还包含bounding box信息，则对每个bounding box序列进行类似的padding。

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
               # 这段代码是一个Python代码片段，用于将数据集中的batch转换为PyTorch张量。该代码使用了PyTorch库中的tensor函数，以及Python中的isinstance函数。
               # 具体来说，这段代码遍历了batch中的所有键值对，并且对值进行了转换。
               # 对于值v，如果它是一个列表，且该列表的第一个元素也是一个列表，则将它转换为PyTorch的int64数据类型张量，并将其保存在一个新的字典中。
               # 否则，如果值v不是一个列表，或者是一个空列表，则直接将其保存在新的字典中。
               # 值得注意的是，这段代码使用了Python中的isinstance函数，判断v的第一个元素是否是一个列表。
               # 这是为了确保对于列表中的嵌套列表，也能够正确地进行转换。
               # 如果不进行这个判断，可能会出现类型转换错误的情况。另外，由于数据类型是int64，因此需要在转换时指定数据类型为int64。
        if has_image_input:
            batch["image"] = image
        return batch
        # 代码中的第一行判断输入的数据是否包含图像数据。如果has_image_input为True，则说明输入数据包含图像数据，将其添加到batch中。
        # 这里假设image是一个PyTorch张量，将其直接存储在batch字典中的 "image"键下。
        # 第二行代码直接返回了整个batch字典。由于前面的代码已经将所有的输入数据都添加到了batch中，因此在返回的过程中，可以确保整个batch中包含了所有的输入数据。
