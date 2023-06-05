from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from transformers.file_utils import ModelOutput


@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None
        # 这段代码定义了一个名为ReOutput的类，该类继承了ModelOutput类。ReOutput类有以下属性：
        # loss：一个可选的浮点数张量，表示模型的损失值。
        # logits：一个浮点数张量，表示模型的输出。这通常是一个用softmax函数转换后的概率分布，表示每个可能的输出标签的概率。
        # hidden_states：一个可选的元组，包含了一个或多个隐藏状态张量，这些张量通常是来自模型的中间层或某些特征提取器。
        # attentions：一个可选的元组，包含了一个或多个注意力张量，这些张量通常是来自模型的自注意力机制。
        # entities：一个可选的字典，包含了模型对输入文本中实体的识别和标注结果。
        # relations：一个可选的字典，包含了模型对输入文本中关系的识别和标注结果。
        # pred_relations：一个可选的字典，包含了模型对输入文本中关系的预测结果。
        # 这些属性的具体含义和用途取决于具体的应用场景和模型设计。