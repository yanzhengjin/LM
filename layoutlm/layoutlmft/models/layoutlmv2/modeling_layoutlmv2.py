# coding=utf-8
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import detectron2
from detectron2.modeling import META_ARCH_REGISTRY
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMIntermediate as LayoutLMv2Intermediate
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOutput as LayoutLMv2Output
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as LayoutLMv2Pooler
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMSelfOutput as LayoutLMv2SelfOutput
from transformers.utils import logging

from ...modules.decoders.re import REDecoder
from ...utils import ReOutput
from .configuration_layoutlmv2 import LayoutLMv2Config
from .detectron2_config import add_layoutlmv2_config


logger = logging.get_logger(__name__)

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlmv2-base-uncased",
    "layoutlmv2-large-uncased",
]


LayoutLMv2LayerNorm = torch.nn.LayerNorm


class LayoutLMv2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
        # 这段代码定义了一个名为LayoutLMv2Embeddings的PyTorch神经网络模块（nn.Module），用于将输入的token IDs和位置信息转换为嵌入向量。
        # 具体来说，这个模块实现了LayoutLMv2模型中的嵌入层，包括以下步骤：
        # 将输入的token IDs转换为对应的词嵌入向量，通常使用预训练的词向量表。
        # 将输入的位置信息转换为位置嵌入向量，通常使用正弦余弦函数来编码位置信息。
        # 将词嵌入向量和位置嵌入向量进行拼接，形成最终的输入嵌入向量。
        # LayoutLMv2Embeddings模块通常作为LayoutLMv2模型的第一个组件，接收token IDs和位置信息作为输入，并输出对应的嵌入向量。

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMv2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            # 这段代码定义了LayoutLMv2Embeddings模块的初始化方法。它接收一个config对象作为输入，其中包含了模型的各种配置参数，如词表大小、嵌入维度、位置编码的最大长度、正则化参数等。
            # 在初始化方法中，首先调用了父类nn.Module的初始化方法，然后定义了一系列嵌入层，包括词嵌入、位置嵌入、坐标嵌入、形状嵌入和类型嵌入等。这些嵌入层用于将输入的token IDs、位置信息和类型信息转换为对应的嵌入向量。
            # 接下来定义了LayerNorm层和Dropout层，用于在嵌入层之后进行正则化和随机失活，以提高模型的泛化能力。
            # 最后，调用了register_buffer方法来注册一个名为"position_ids" 的缓存张量，用于缓存位置编码的索引值，可以有效地减少重复计算。
            # 具体来说，这个缓存张量是一个一维的张量，长度为max_position_embeddings，包含了从0到max_position_embeddings - 1
            # 的连续整数。在模型的前向传播过程中，这个缓存张量将被用于获取位置编码的索引值，以提高位置编码的计算效率。

    def _cal_spatial_position_embeddings(self, bbox):
        #这段代码定义了一个名为_cal_spatial_position_embeddings的方法，用于计算空间位置信息的嵌入向量。
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings
        # 具体来说，这个方法接收一个名为bbox的三维张量作为输入，其中第一维表示batch size，第二维表示token序列的长度，第三维包含了每个token对应的bounding box信息。
        # bbox张量的形状为(batch_size, seq_length, 4)，其中4表示左上角和右下角的坐标值。
        # 在计算嵌入向量时，该方法首先通过x_position_embeddings和y_position_embeddings嵌入层分别计算左上角和右下角的x、y坐标位置的嵌入向量，
        # 然后通过h_position_embeddings和w_position_embeddings嵌入层计算bbox的高度和宽度的嵌入向量。
        # 最后，这个方法将左上角、右下角、高度和宽度的嵌入向量拼接在一起，形成最终的空间位置嵌入向量，返回给调用者。
        # 这个空间位置嵌入向量将作为输入嵌入向量的一部分，参与后续的模型计算。

class LayoutLMv2SelfAttention(nn.Module):
    #这段代码定义了一个名为LayoutLMv2SelfAttention的类，它继承自nn.Module类，用于实现LayoutLMv2模型中的自注意力机制。
    def __init__(self, config):
        super().__init__()
        #这段代码定义了一个名为LayoutLMv2SelfAttention的类，它继承自nn.Module类，用于实现LayoutLMv2模型中的自注意力机制中的注意力计算部分。
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 在该类的初始化方法中，根据输入的config对象初始化了注意力机制所需的各个参数，包括注意力头数、隐藏层大小等等。
        # 如果hidden_size不是attention_head_size的倍数，并且config对象没有embedding_size属性，将抛出ValueError异常。
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            # 如果config.fast_qkv为True，那么使用一个线性层self.qkv_linear，将原始嵌入向量转换成query、key、value三个部分，
            # 并使用self.q_bias和self.v_bias两个参数向量作为query和value的偏置；
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            #否则分别使用query、key、value三个线性层对原始嵌入向量进行转换。
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # transpose_for_scores是LayoutLMv2SelfAttention类中的一个方法，用于将输入张量进行变形以适应多头注意力机制的计算。
        # 具体而言，该方法将输入张量x变形为形状为[batch_size, sequence_length, num_attention_heads, attention_head_size]的张量，
        # 然后对最后两个维度进行转置，即将维度2和3交换，得到形状为[batch_size, num_attention_heads, sequence_length, attention_head_size]的输出张量，用于后续的多头注意力机制的计算。
        # 这样做的目的是将输入张量中代表不同特征维度的维度信息（如词向量维度、位置编码维度、边界框编码维度等）分割成多个头，每个头单独计算注意力，提高模型的表示能力和泛化能力。

    def compute_qkv(self, hidden_states):
        # 这段代码实现了计算query、key、value的功能，用于self - attention的计算。
        # 如果使用的是fast_qkv方式，即使用线性层一次计算query、key、value，同时使用可学习的偏置项；
        # 否则，使用三个不同的线性层计算query、key、value。最终返回计算出的query、key、value。
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        # 具体地说，如果self.fast_qkv为真，表示要使用快速的方法计算qkv矩阵。在这种情况下，qkv的计算是通过一个线性层 self.qkv_linear对输入的hidden_states进行变换得到的，然后通过torch.chunk将其拆分成三部分q, k, v。
        # 接着，通过torch.chunk拆分出的q和v，加上偏置项self.q_bias和 self.v_bias，然后进行注意力计算。
        # 需要注意的是，因为q, k, v 在batch_size和sequence_length两个维度上都有值，而 q_bias 和v_bias只有在batch_size维度上有值，所以在加上偏置项时需要进行维度转换。
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
            # 这段代码是定义了一个forward函数，它是一个pytorch中的模型的前向传播函数，用于对输入进行处理并输出预测结果。
            # 函数接受多个输入参数，包括：hidden_states: 表示输入的特征向量，即模型的输入
            # attention_mask: 表示输入的mask矩阵，用于控制模型在处理输入时忽略一些无效的信息
            # head_mask: 表示对不同的attention
            # head进行掩码处理的向量
            # encoder_hidden_states: 表示编码器的输出，用于实现一些attention机制
            # encoder_attention_mask: 表示编码器输出的mask矩阵，也用于实现一些attention机制
            # past_key_value: 表示前面计算过的key value pairs，用于实现一些attention机制
            # output_attentions: 是否输出注意力权重
            # rel_pos: 表示相对位置编码
            # rel_2d_pos: 表示二维相对位置编码
            # 这些参数的具体含义可以参考模型的文档或者函数的调用部分来理解。
        q, k, v = self.compute_qkv(hidden_states)
        # 具体来说，这里调用了compute_qkv方法，将hidden_states通过三个独立的线性变换映射到不同的空间中，分别得到query、key和value表示，即q、k和v。
        # 这是自注意力机制中的一个重要步骤，用于计算每个token之间的注意力权重。
            # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
            # 这段代码出现在Transformer模型的Multi - Head Attention中，其目的是将输入的query、key和value经过线性变换后转换为score矩阵的形式。
            # 具体来说，query、key和value都是形状为(batch_size, sequence_length, hidden_size) 的张量，其中hidden_size是词嵌入的维度。
            # 为了将query、key和value输入到Multi - Head Attention中，需要首先通过三个线性层，将它们分别映射到一个维度为hidden_size的中间表示。
            # 然后，对这些中间表示进行reshape，使其变成(batch_size,num_attention_heads,sequence_length,attention_head_size)的形式，
            # 其中num_attention_heads表示分头注意力的头数，attention_head_size表示每个头的维度。
            # 之后，再将query、key和value分别传入到Multi - Head Attention中，用来计算score矩阵。
        query_layer = query_layer / math.sqrt(self.attention_head_size)
            # 在这段代码中，self.transpose_for_scores(q)、self.transpose_for_scores(k)和self.transpose_for_scores(v)的作用是将query、key和value的维度从(batch_size, sequence_length, hidden_size)
            # 转换为(batch_size, num_attention_heads, sequence_length, attention_head_size)，
            # 其中num_attention_heads和attention_head_size分别由模型的参数设置。
            # 在计算score矩阵时，需要将query和key的转置相乘，所以为了避免结果过大，还需要将query_layer除以math.sqrt(self.attention_head_size)。
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # 这行代码的作用是计算query和key之间的点积注意力得分矩阵，具体来说，它将query_layer和key_layer矩阵相乘，然后进行矩阵转置，
            # 以便后续的softmax操作，即将query_layer的每个向量与key_layer的每个向量做点积，得到每个query与key的得分。
            # 最终得到的attention_scores矩阵的大小是[batch_size, num_heads, seq_length, seq_length]，其中每个元素表示每个query在seq_length个key上的得分。
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        #这段代码中，首先判断模型是否包含相对位置编码，如果有，就将其加到注意力矩阵上。
        #接着判断模型是否包含2D相对位置编码，如果有，也将其加到注意力矩阵上。这两种相对位置编码的作用是帮助模型学习到更加精确的位置信息。
        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
            # 这两行代码用于计算注意力概率。其中，attention_scores是注意力得分矩阵，经过mask之后，将其将不需要计算注意力的位置填充为负无穷，避免这些位置对后续计算产生影响。
            # 然后，使用softmax函数将注意力得分矩阵转化为注意力概率矩阵attention_probs。
            # softmax函数可以将注意力得分矩阵中的值转化为0 - 1之间的概率分布，使得所有值相加等于1，从而能够更好地描述输入序列中各个位置的重要程度。
            # 这里使用PyTorch的F.softmax函数进行计算。
            # 注意，为了避免精度问题，需要先将attention_scores转换为浮点型float()，然后使用type_as()方法将其转换为与value_layer相同的数据类型。
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
            # 这段代码是多头注意力机制的核心部分，计算出上下文向量。首先将前面计算出的注意力分数张量和V矩阵进行矩阵乘法，得到每个头部的上下文向量。
            # 然后通过permute函数交换张量的维度顺序，将头部和序列长度两个维度交换，最后通过view函数将张量的形状改为(batch_size, seq_len, hidden_size)。
            # 这里的 all_head_size即等于num_attention_heads * attention_head_size。

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    # 这段代码是一个PyTorch模型的正向传递（forward）方法，用于实现多头自注意力机制。
    # 具体地，这个方法接受一些输入张量，包括hidden_states表示输入的序列，attention_mask表示序列中需要mask掉的部分，head_mask表示需要mask掉的注意力头，
    # encoder_hidden_states表示另一个序列的隐藏状态（一般是encoder的输出），encoder_attention_mask表示另一个序列的mask，
    # past_key_value表示已有的键值对（在decoder中使用），以及两个表示位置信息的张量rel_pos和rel_2d_pos。
    # 这个方法首先使用一个线性变换将输入序列映射为一个query、key和value张量，然后分别对它们进行多头切分（即将这些张量按注意力头的数量进行分块）。
    # 接着，这个方法将query和key张量进行矩阵乘法得到注意力分数张量，将这个张量与attention_mask相乘，然后进行softmax归一化。
    # 接着，将这个归一化的张量与value张量进行加权求和，得到最终的上下文向量。
    # 最后，这个方法返回这个上下文向量和attention_probs（如果设置了输出注意力分数）作为输出。


class LayoutLMv2Attention(nn.Module):
        # 这段代码定义了一个名为 "LayoutLMv2Attention" 的PyTorch模块类，它是深度学习中的注意力机制模块。
        # 注意力机制是一种机制，通过计算不同元素之间的相关性来将一组元素编码成另一组元素，通常在序列到序列（seq2seq）任务和自然语言处理（NLP）任务中使用。
        # 在这个模块中，它可能用于对输入中的布局信息进行编码和提取，以便在布局分析或图像识别等任务中使用。
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)
        self.pruned_heads = set()
        # 这段代码定义了一个名为"LayoutLMv2Attention"的PyTorch模块类的构造函数 "init"，它接受一个 "config"参数。
        # 在构造函数中，它首先调用父类的构造函数"super().init()"来初始化父类中的一些属性。
        # 然后，它定义了三个属性：
        # self：一个"LayoutLMv2SelfAttention"类型的模块实例，该模块可能是用于对输入中的布局信息进行编码和提取的注意力机制模块。
        # output：一个"LayoutLMv2SelfOutput"类型的模块实例，该模块可能是用于将自注意力模块的输出进行处理的模块。
        # pruned_heads：一个集合，用于记录在模型中被修剪（pruned）掉的注意力头（attention head）的索引。
    def prune_heads(self, heads):
        # 这个函数的作用是剪枝self - attention中的头部（attention heads），以减少模型参数和计算量。
        # 参数heads是一个需要保留的头部编号列表，将从原来的头部中移除其余的头部。
        # 具体来说，该函数会调用辅助函数find_pruneable_heads_and_indices找到需要被剪枝的头部及其在模型中的位置，然后将这些头部从self.pruned_heads列表中移除。
        if len(heads) == 0:
            return
        #如果传入要删除的头部列表为空，那么直接返回，不进行下面的操作。
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 这里调用了一个函数find_pruneable_heads_and_indices，它用于找到可以剪枝的注意力头，并返回它们的下标。
        # 具体来说，该函数会输入剪枝头列表，注意力头的数量，每个注意力头的维度以及已经剪枝的头信息，然后返回可以剪枝的头以及对应的下标。
        # 返回的头部下标将在接下来的剪枝操作中使用。

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 这段代码执行了一个"prune_linear_layer"函数，用于修剪线性层的权重矩阵。
        # 具体来说，它针对"self"属性中的三个线性层（"query"、"key"、"value"）和"output"属性中的一个线性层（"dense"），调用"prune_linear_layer"函数进行修剪。
        # 其中，"index"参数是一个整数列表，表示要修剪的权重矩阵中要保留的维度的索引。这些索引对应的维度将被保留，其他维度将被修剪掉。
        # "prune_linear_layer"函数返回修剪后的权重矩阵，并将修剪后的权重矩阵赋值回原来的属性中。
        # 在"output.dense"的修剪中，还额外指定了"dim=1"参数，表示修剪的是权重矩阵的第一维，即输入维度。
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        # 这段代码更新了模型的超参数，同时记录被修剪掉的注意力头的索引。
        # 具体来说，它首先计算了修剪后剩余的注意力头数量，即"num_attention_heads - len(heads)"，并将其赋值回"self.self.num_attention_heads"属性中，从而更新了模型的注意力头数量。
        # 然后，它计算了修剪后每个注意力头的大小，即"attention_head_size * num_attention_heads"，并将其赋值回"self.self.all_head_size"属性中，从而更新了模型每个注意力头的大小。
        # 最后，它将被修剪掉的注意力头的索引，即"heads"参数中的值，添加到 "self.pruned_heads"属性中，从而记录了被修剪掉的注意力头的索引。
        # 这个属性在模型中其他部分可能会用到，例如在前向传递时跳过被修剪掉的注意力头。

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 这段代码定义了一个名为"forward"的PyTorch模块方法，用于对输入进行前向传递计算。
        # 具体来说，这个方法接受以下参数：
        # "hidden_states"：输入的特征表示，可以是一个三维的张量，形状为[batch_size, sequence_length, hidden_size]。
        # "attention_mask"：用于指示哪些位置需要注意力计算，可以是一个二维的张量，形状为[batch_size, sequence_length]，其中值为0的位置表示需要进行注意力计算，值为1的位置表示不需要进行注意力计算。
        # "head_mask"：用于指定哪些注意力头需要保留，可以是一个列表，长度为模型中的注意力头数量，其中值为0的位置表示需要被修剪掉的注意力头，值为1的位置表示需要保留的注意力头。
        # "encoder_hidden_states"：用于进行编码器 - 解码器注意力计算的编码器隐藏状态，可以是一个三维的张量，形状为[batch_size, sequence_length, hidden_size]。
        # "encoder_attention_mask"：用于指示哪些编码器隐藏状态需要注意力计算的掩码张量，可以是一个二维的张量，形状为[batch_size, sequence_length]，其中值为0的位置表示需要进行注意力计算，值为1 的位置表示不需要进行注意力计算。
        # "past_key_value"：用于存储过去计算过的键 - 值对，可以是一个元组，其中第一个元素为存储过去计算过的键，第二个元素为存储过去计算过的值。
        # "output_attentions"：用于指示是否返回注意力权重，如果设置为True，则返回一个四元组，其中第四个元素为注意力权重，否则返回一个三元组。
        # "rel_pos"：用于提供相对位置编码信息的张量，可以是一个二维的张量，形状为[sequence_length, sequence_length]。
        # "rel_2d_pos"：用于提供二维相对位置编码信息的张量，可以是一个四维的张量，形状为[num_heads, window_size, window_size, dim_per_head]。
        # 这个方法的功能是对输入进行自注意力计算，并根据注意力权重对输入进行加权求和，然后通过一个前馈网络进行处理，最后返回处理后的结果。在计算自注意力时，可以使用相对位置编码信息和编码器 - 解码器注意力。
        # 如果需要修剪注意力头，则根据"head_mask"参数进行修剪。如果需要返回注意力权重，则将注意力权重作为额外的输出返回。
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 这段代码调用了"self"属性的 "self"方法，对输入进行自注意力计算，得到输出"self_outputs"。
        # 具体来说，这个方法接受了和上面介绍的"forward"方法一样的参数，并根据这些参数对输入进行自注意力计算，并返回一个四元组或三元组，
        # 其中第一个元素为注意力加权后的特征表示，第二个元素为注意力权重，第三个元素为过去计算过的键 - 值对，第四个元素只在 "output_attentions" 设置为True时返回，表示注意力权重。
        # 注意，这个方法中并没有使用"self"关键字，而是使用了 "self.self"属性，因为"self"关键字已经在类定义中被占用了。
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 这段代码将自注意力计算得到的注意力加权特征表示"self_outputs[0]"和输入"hidden_states"作为输入，调用了"output"方法，得到输出"attention_output"。
        # 然后将"attention_output"和自注意力计算方法返回的其他元素（如注意力权重等）组成一个元组"outputs"，其中如果设置"output_attentions" 为True，则会将注意力权重作为元组的第二个元素返回。
        # 需要注意的是，这里的"self.output"是在"LayoutLMv2Attention"类中定义的一个"LayoutLMv2SelfOutput"类的实例对象，用来将自注意力计算的结果进行处理。
        return outputs
        #将上面得到的"outputs"元组作为方法的返回值返回。因此，当调用这个"forward"方法时，将会得到一个包含注意力加权特征表示和其他可能的信息的元组，这个元组的具体元素数目和类型取决于模型的配置和输入。


class LayoutLMv2Layer(nn.Module):
    # 定义了一个名为"LayoutLMv2Layer"的类，该类继承自PyTorch的nn.Module类。这个类表示LayoutLMv2模型的一个单层，它包含以下组件：
    # 一个自注意力层（LayoutLMv2Attention），用于对输入进行自注意力计算。
    # 一个前馈网络层（LayoutLMv2Intermediate），用于将自注意力计算的结果映射到高维空间。
    # 一个输出层（LayoutLMv2Output），用于将前馈网络层的输出进行处理，得到该层最终的输出结果。
    # 该类的实例化需要传入一个配置参数"config"，该参数包含了该层的超参数和其他配置信息。
    # 该类中的"forward"方法接受一个输入张量"hidden_states"，以及其他可选参数，然后依次对输入进行自注意力计算、前馈网络计算和输出层处理，最终返回该层的输出结果。
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 代码定义了一个"LayoutLMv2Layer"类的构造函数"init"，该函数接受一个"config"参数作为输入，用于初始化该层的配置参数和组件。
        # 具体来说，该构造函数首先调用父类nn.Module的构造函数，然后根据传入的"config"参数，初始化了以下成员变量：
        # "chunk_size_feed_forward"：前馈网络层的块大小，用于控制前馈网络的计算量和内存消耗。
        # "seq_len_dim"：输入张量中表示序列长度的维度。默认为1，即输入张量的第二个维度。
        # "attention"：该层的自注意力层（LayoutLMv2Attention）实例。
        # "is_decoder"：该层是否作为解码器使用的标志。
        # "add_cross_attention"：是否添加交叉注意力的标志，用于实现多头注意力机制。
        # 这些成员变量将在该层的forward方法中使用。
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)
        # 实现了一个"LayoutLMv2Layer"类的构造函数"init"的另一部分。
        # 如果 "add_cross_attention"参数为True，表示需要添加交叉注意力机制，此时必须将该层作为解码器使用（即"is_decoder"参数为True），否则会抛出异常。
        # 如果"add_cross_attention"为 False，则不需要添加交叉注意力机制。
        # 无论如何，该构造函数都会初始化以下组件：
        # "intermediate"：前馈网络层（LayoutLMv2Intermediate）实例，用于将自注意力计算的结果映射到高维空间。
        # "output"：输出层（LayoutLMv2Output）实例，用于将前馈网络层的输出进行处理，得到该层最终的输出结果。
        # 如果需要添加交叉注意力机制，还会额外初始化一个交叉注意力层（LayoutLMv2Attention）实例，用于计算输入张量与编码器输出张量之间的交叉注意力。

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 实现了一个"LayoutLMv2Layer"类的前向传播函数"forward"，用于计算该层的输出。
        # 该函数接收一些输入参数：
        # "hidden_states"：输入张量，shape为(batch_size, sequence_length, hidden_size)。
        # "attention_mask"：用于屏蔽无关的输入的掩码张量，shape为(batch_size, sequence_length)或(batch_size, 1, 1, sequence_length)。
        # "head_mask"：用于屏蔽某些注意力头的掩码张量，shape为(num_attention_heads, )或(num_hidden_layers, num_attention_heads)。
        # "encoder_hidden_states"：编码器的输出张量，用于计算交叉注意力机制。shape为(batch_size, encoder_sequence_length, hidden_size)。
        # "encoder_attention_mask"：编码器的注意力掩码张量，用于计算交叉注意力机制。shape为(batch_size, 1, 1, encoder_sequence_length)。
        # "past_key_value"：过去的注意力密钥值对，用于支持BERT等模型的
        # "memorization"机制。它是一个元组(key, value)，每个元素的shape为(batch_size, num_attention_heads, past_sequence_length, head_dim)。
        # "output_attentions"：是否需要输出注意力权重。
        # "rel_pos"：相对位置编码张量，shape为(batch_size, num_attention_heads, sequence_length, sequence_length)。
        # "rel_2d_pos"：二维相对位置编码张量，shape为(batch_size, num_attention_heads, sequence_length, sequence_length, 2)。
        # 在函数内部，首先通过自注意力计算层（self.attention）计算注意力得分，并将结果输入到前馈网络层（self.intermediate）中进行处理，最后再通过输出层（self.output）进行处理，得到最终的输出张量。
        # 如果 "add_cross_attention"为True，则还需要计算交叉注意力。
        # 这时，需要将输入张量和编码器输出张量作为输入，分别通过自注意力计算层（self.attention）和交叉注意力计算层（self.crossattention）进行计算，
        # 再将两个结果进行加权平均处理，并通过前馈网络层和输出层处理，得到最终的输出张量。
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 这行代码的作用是将 past_key_value 变量拆分成前两个元素并赋值给 self_attn_past_key_value，如果 past_key_value 为空则将其赋值为 None。
        # 根据代码上下文来看，这里的 past_key_value 是用来缓存自注意力机制中的键值对的。
        # 在训练过程中，为了提高效率，模型会将中间结果保存在  past_key_value 中，以便在下一次前向传递时复用。
        # 这里将 past_key_value 变量拆分为前两个元素，是为了将其传递给自注意力模块进行缓存。
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 作用是调用 self.attention模块的前向传递函数forward进行自注意力计算。其中传入的参数包括：
        # hidden_states: 输入序列的特征表示，形状为(batch_size, sequence_length, hidden_size)。
        # attention_mask: 注意力掩码，形状为(batch_size, sequence_length)或(batch_size, 1, 1, sequence_length)，用于指示哪些位置需要被忽略。
        # head_mask: 注意力头掩码，形状为(num_attention_heads, ) 或(num_hidden_layers, num_attention_heads)，用于指定哪些注意力头需要被屏蔽。
        # output_attentions: 是否返回注意力权重，如果为True，则会在输出中包含注意力权重。
        # past_key_value: 自注意力中缓存的键值对，用于加速计算，形状为(2, batch_size, num_heads, past_sequence_length, head_size)。
        # rel_pos: 相对位置编码，形状为(batch_size, num_heads, sequence_length, past_sequence_length)，用于处理序列中位置信息的相关性。
        # rel_2d_pos: 二维相对位置编码，形状为(batch_size, num_heads, sequence_length, sequence_length)，用于处理序列中二维位置信息的相关性。
        # 返回值self_attention_outputs包含以下元素：
        # attention_output: 自注意力的输出特征表示，形状同输入的hidden_states。
        # next_self_attention_past_key_value: 下一个时间步的自注意力中缓存的键值对，用于加速计算，形状同输入的past_key_value。
        # self_attention_weights（如果output_attentions = True）：自注意力的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)。
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
        #意思是，如果模型是decoder并且有encoder_hidden_states作为输入，则必须设置config.add_cross_attention = True来实例化带有跨注意力层的模型，否则会抛出错误提示。
            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 主要是实现encoder - decoder注意力机制。如果当前层是decoder层并且encoder_hidden_states不为空，则需要进行encoder - decoder注意力机制，即使用encoder_hidden_states对attention_output进行交叉注意力计算。
            # 这里通过判断是否有self.crossattention来确定是否有进行encoder - decoder注意力计算的操作。
            # 如果有，就使用self.crossattention计算交叉注意力，得到cross_attention_outputs。
            # cross_attn_past_key_value参数用于存储过去的键值对，即过去时间步骤中计算的encoder - decoder注意力的结果。
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights
            # 实现了在Decoder中使用Encoder hidden states进行跨层Attention的功能。
            # 如果encoder_hidden_states不是None且self.is_decoder为True，则需要使用cross - attention。
            # 所以，首先会检查是否已经有crossattention模块，如果没有则会报错。
            # 然后会提取已有的past_key_value参数中最后两个参数（因为crossattention需要用到），如果没有则设置为None。
            # 接着调用crossattention模块对attention_output和encoder_hidden_states进行attention计算，并将计算结果拼接到之前的outputs变量中。
            # 最终，返回attention_output和outputs（如果有output_attentions则也返回attentionweights）。
            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
            # 实现了cross - attention机制，即对于decoder中的每一个token，都会对encoder的hidden state进行attention，获取对于当前token的上下文信息。
            # 如果当前层需要进行cross - attention操作，则会检查是否实例化了crossattention模型，并将crossattention模型的输出结果与selfattention模型的输出结果合并作为本层的最终输出结果。
            # 其中，present_key_value用于存储当前时刻的query、key和value向量，以便于下一次的attention计算，是一个元组类型，
            # 包括两个张量，每个张量的shape均为( batch_size, num_heads, sequence_length, head_size)。
            # 而cross_attn_present_key_value则是执行crossattention模型后的query、key和value向量，与present_key_value相同，用于下一次的attention计算。
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        # 这段代码是将attention_output传入feed_forward_chunk进行前向计算，因为当序列很长时，一次性计算可能会导致内存不足，所以这里采用了分块计算的方法。
        # apply_chunking_to_forward函数就是用来实现分块计算的，它会将attention_output在第1维（即序列长度）上进行分块，每块的长度是self.chunk_size_feed_forward，
        # 然后对每一块都调用self.feed_forward_chunk进行计算，最后将每块的输出拼接起来得到整个序列的输出layer_output。
        # 然后将layer_output和之前的outputs合并为一个元组返回。
        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        # 这里的 present_key_value是指上下文编码器对应的历史的key - value对，用于在decoder端的self - attention过程中做缓存加速，以便实现自回归式的文本生成任务。
        # 如果是decoder模型，则将此key - value对存储到输出中。
        return outputs
        # 这是 LayoutLMv2Layer 类中的forward方法的返回值。它返回一个元组，其中包含经过当前层处理后的隐藏状态，和可能的注意力权重（如果output_attentions参数设置为True），
        # 以及可选的跨层注意力的键 / 值状态（仅适用于解码器模型）。这些返回值将成为LayoutLMv2Encoder或LayoutLMv2Decoder类中的下一层的输入。
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # 实现了LayoutLMv2Layer类中的feed_forward_chunk方法，用于对self-attention输出进行前馈计算。
        # 具体实现过程是：首先通过intermediate层对输入进行线性变换并应用激活函数，然后再将结果输入到output层进行dropout、残差连接和线性变换得到最终的输出。
        # 最终的输出是下一层的输入。

def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    # 用于将相对位置编码为桶的索引。它接受相对位置的张量（可以是正的或负的），并将其离散化为固定数量的桶中的一个。
    # 函数使用的离散化方法是将相对位置除以一个预定义的最大距离，然后将结果限制在一个预定义的范围内（默认为 - 128到128）。
    # 离散化后的结果用于在注意力机制中使用相对位置编码。
    # 函数还允许指定是否使用双向编码，以及要使用的桶的数量。
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # 作用是计算相对位置编码中的桶编号。如果bidirectional为True，则将可正可负的相对位置分为两个区间，每个区间有num_buckets // 2个桶；
    # 如果为False，则只考虑正的相对位置。
    # 然后根据相对位置的正负，将桶编号加上一个偏移量，即可确定最终的桶编号。
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact
        #max_exact是一个整数值，表示桶的一半数量。
        # 如果相对位置的绝对值小于max_exact，则将其分配到一个固定的桶中，否则将其分配到一个自适应桶中。
        # is_small是一个布尔值张量，表示相对位置是否小于max_exact。
    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    # 作用是，计算相对位置编码的桶id。首先，将输入的相对位置值取绝对值，然后如果是双向的，将正负相对位置分别映射到桶中的不同位置。
    # 接下来，计算相对位置的绝对值是否小于最大精确值，如果是，则直接将相对位置映射到id为相对位置值的桶；
    # 否则，根据相对位置和最大距离之间的比例计算相对位置的桶id。这里的val_if_large变量即为计算得到的桶id。
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    # 意思是，如果计算出来的桶的数值大于等于num_buckets，那么就将其置为num_buckets - 1。
    # 因为桶的数值最大是num_buckets - 1，超过了也要保持在这个范围内。
    # 使用torch.min()实现这个操作。
    ret += torch.where(is_small, n, val_if_large)
    # 实现了相对位置编码的分桶操作。其中，relative_position表示两个位置之间的相对距离，bidirectional表示是否为双向编码，num_buckets表示编码的桶数，max_distance表示能够编码的最大距离。
    # 具体地：如果是双向编码，将桶数除以2，然后根据relative_position的正负性选择对应的一半桶。
    # 如果相对距离在max_exact以内，则直接使用相对距离作为桶的编号。
    # 如果相对距离超过max_exact，则使用对数函数将其映射到余下的桶中，然后将桶的编号加上已有编号。
    # 如果桶的编号超过num_buckets，则将其设为num_buckets - 1。
    # 最后，将结果加到ret中，并返回ret。
    return ret
    #用于将相对位置编码映射到离散的桶中


class LayoutLMv2Encoder(nn.Module):
    # 它是LayoutLMv2的编码器模块。LayoutLMv2是一个文档级别的多模态语言模型，它能够同时对文本和布局信息进行建模，用于文档分析、信息提取、问答等任务。
    # 该编码器模块包括多个子模块，如自注意力层、前馈全连接层、交叉注意力层等，用于从输入的文本和布局信息中提取高级语义特征。
    def __init__(self, config):
        # 定义了 LayoutLMv2Encoder类的构造函数，用于实例化 LayoutLMv2Encoder类的对象。
        # 信息在函数中，config参数是一个配置类对象，包含了该模型的所有超参数和配置。
        # 函数中首先调用父类nn.Module的构造函数来初始化LayoutLMv2Encoder类的实例。
        # 然后，它使用配置信息中的参数来定义一些类的实例变量和子模块。这些子模块包括嵌入层、Transformer编码器、位置编码器和 LayerNorm层等。
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])
        # 定义了一个 LayoutLMv2Encoder类，它继承了nn.Module类。在这个类的构造函数中，首先调用 nn.Module类的构造函数进行初始化，然后保存了传入的config 参数。
        # 最后，定义了一个nn.ModuleList 对象self.layer，包含了config.num_hidden_layers 个LayoutLMv2Layer模块。
        # 因此，LayoutLMv2Encoder是由多个 LayoutLMv2Layer组成的，用于实现LayoutLMv2的编码器。
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        # 这两个属性是用来标记该Encoder是否有相对位置编码和空间位置编码的。
        # 在初始化函数中，将config中的has_relative_attention_bias和has_spatial_attention_bias赋值给了self.has_relative_attention_bias和self.has_spatial_attention_bias。
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)
        #判断模型中是否有相对位置注意力偏置和空间注意力偏置，并根据有无相对位置注意力偏置做出不同的处理。
        # 如果有相对位置注意力偏置，则需要定义相对位置编码的相关参数，并使用线性层对编码进行处理。
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
        #在初始化LayoutLMv2Encoder时，判断是否需要使用相对位置注意力偏置和空间注意力偏置。
        # 如果需要使用相对位置注意力偏置，则根据相关参数建立一个线性层self.rel_pos_bias；
        # 如果需要使用空间注意力偏置，则根据相关参数建立两个线性层self.rel_pos_x_bias和self.rel_pos_y_bias。这些线性层都将用于计算注意力偏置。
    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        # 作用是计算一维位置嵌入（Position Embedding）矩阵，用于在输入序列中区分不同位置的标识符。
        # 它接受两个参数：hidden_states是形状为[batch_size, seq_length, hidden_size] 的张量，表示输入序列的隐藏状态；position_ids是形状为[batch_size, seq_length]的张量，表示输入序列中每个标识符的位置编号。
        # 函数的返回值是形状为[batch_size, seq_length, hidden_size]的张量，表示输入序列中每个位置对应的位置嵌入向量。
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # 作用是计算相对位置矩阵，其中position_ids.unsqueeze(-2)表示将position_ids张量在倒数第二个维度上新增一个维度，
        # 这样可以将position_ids张量变成一个形状为(batch_size, seq_len, 1)的张量；position_ids.unsqueeze(-1)
        # 同理可以将position_ids张量变成一个形状为(batch_size, 1, seq_len)的张量。
        # 然后使用两个张量进行减法运算，得到的就是形状为(batch_size, seq_len, seq_len)的相对位置矩阵。
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 作用是将二维位置矩阵转化为相对位置桶（relative position bucket）表示。
        # 其中，relative_position_bucket函数会将每个相对位置映射为其所属的位置桶（bucket），这个函数可以理解为一种类似于哈希函数的操作，它将相对位置映射到 $[0, num_buckets - 1]$ 的整数区间。
        # 在这里，rel_pos_mat 是一个形状为[batch_size, seq_length, seq_length]的张量，表示每对位置之间的相对距离，num_buckets表示相对位置桶的数量，max_distance表示相对距离的最大值。
        # 经过relative_position_bucket函数处理后，rel_pos的形状仍为[batch_size, seq_length, seq_length]，其中每个元素都表示一个相对位置所属的位置桶编号。
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        # 是用于计算相对位置嵌入矩阵的逻辑。首先通过 relative_position_bucket()函数将相对位置rel_pos_mat映射到一个固定大小的桶中，得到整数编码rel_pos。
        # 接着，利用F.one_hot() 函数将整数编码转化为one - hot编码，得到rel_pos的one - hot编码rel_pos_onehot。
        # 然后将rel_pos_onehot输入到一个线性层self.rel_pos_bias中，得到rel_pos的相对位置嵌入rel_pos_emb。
        # 最后将rel_pos_emb调整形状，使其符合注意力矩阵的维度要求，并返回rel_pos_emb。这些相对位置嵌入将用于后续的注意力计算。
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        # 这是一个计算二维位置嵌入（position embedding）的函数。其中，输入参数hidden_states是一个形状为[batch_size, seq_length, hidden_size]的张量，
        # 表示编码器的输出，bbox是一个形状为[batch_size, seq_length, 4]的张量，表示输入文本中每个token的边框坐标。
        # 函数首先计算出所有token之间的相对位置向量，并使用relative_position_bucket函数将这些相对位置向量映射为桶ID。
        # 然后，将桶ID转换为one - hot编码，并传递给线性层self.rel_pos_{x, y}_bias，得到相对位置偏置向量。
        # 最后，将相对位置偏置向量沿着通道维度重新排列，使得它的形状变为[ batch_size, num_attention_heads, seq_length, hidden_size // num_attention_heads]，并返回这个张量作为二维位置嵌入。
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        # 将传入的bbox张量沿着第二个维度切片，并选取第0个切片和第3个切片，即bbox张量中每个样本对应的bounding box中左上角和右下角的x, y坐标。
        # 这里假设输入的bbox张量的shape为(batch_size, num_boxes, 4)，其中第二个维度的大小为4，表示每个bounding box的左上角和右下角坐标分别为(xmin, ymin, xmax, ymax)。
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # 是基于输入的bounding box计算相对位置矩阵。position_coord_x是一个形状为(batch_size, seq_length) 的张量，其中的每个元素表示对应token的bounding box的左上角的x坐标。
        # 同样，position_coord_y是一个形状为(batch_size, seq_length)的张量，其中的每个元素表示对应token的bounding box的右下角的y坐标。
        # rel_pos_x_2d_mat和rel_pos_y_2d_mat是根据输入的bounding box计算的相对位置矩阵，分别代表x和y方向的相对位置。
        # 这两个矩阵的形状都是(batch_size, seq_length, seq_length)，其中的每个元素表示两个token之间在x或y方向的相对距离。
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 这里是根据输入的2D坐标位置计算x方向的相对位置嵌入。
        # 通过使用relative_position_bucket()函数将相对位置矩阵rel_pos_x_2d_mat转换为桶索引，然后使用这些桶索引作为输入进行one - hot编码。
        # 最后将one - hot编码的结果传递给self.rel_pos_x_bias线性层进行映射和处理。
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 是为了计算2D位置嵌入，首先从bounding box的左上角和右下角坐标中获取x轴和y轴的坐标值，然后计算出这些坐标之间的相对位置，
        # 再使用relative_position_bucket函数将相对位置分配到不同的bucket中，最终得到相对位置的离散表示rel_pos_x和rel_pos_y。
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        # 这里通过相对位置编码矩阵rel_pos_x_2d_mat和rel_pos_y_2d_mat分别计算x方向和y方向的相对位置编码。
        # 之后，将相对位置编码矩阵通过relative_position_bucket函数映射到相对位置桶中，并转换为one - hot编码。
        # 最后，通过线性层rel_pos_x_bias和rel_pos_y_bias分别将one - hot编码映射为相对位置偏置向量，并对维度进行变换，以便与注意力分数矩阵进行加权操作。
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        # 作用是计算二维位置嵌入（2D position embedding）。
        # 具体地，bbox参数是一个大小为(batch_size, seq_length, 4)的张量，其中第三个维度的4个元素分别表示一个bounding box的左上角和右下角的x和y坐标。
        # 这些坐标用于构建相对位置矩阵，以计算二维相对位置桶（rel_pos_x和rel_pos_y）。
        # relative_position_bucket函数将相对位置矩阵离散化到一个固定数量的桶中，而nn.Linear则将离散化的相对位置桶投影到注意力头数量的维度上，生成2D相对位置嵌入rel_2d_pos。
        # 最后，函数返回rel_2d_pos。
        return rel_2d_pos
        # 计算了二维空间中的位置嵌入。具体来说，它从边界框的坐标计算出相对位置矩阵，对这个矩阵进行量化处理（使用relative_position_bucket函数），然后将其转换成one - hot编码的张量。
        # 接下来，将这个张量输入到两个线性层（rel_pos_x_bias和rel_pos_y_bias）中，得到两个不同的位置嵌入。
        # 最后，将这两个位置嵌入相加，以得到二维位置嵌入，返回它。

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        # 这是一个PyTorch模型的前向传递方法，用于对输入的张量进行一系列的操作，最终得到输出的张量。
        # 具体参数的含义如下： hidden_states: 输入的张量，shape为[ batch_size, sequence_length, hidden_size]，其中batch_size表示batch大小，sequence_length表示序列长度，hidden_size表示隐藏状态的维度。
        # attention_mask: 注意力掩码，用于指示哪些位置需要被屏蔽，shape为[batch_size, 1, 1, sequence_length]。
        # head_mask: 注意力头掩码，用于指示哪些注意力头需要被屏蔽，shape为[num_attention_heads]。
        # encoder_hidden_states: 编码器的隐藏状态，用于进行跨层Attention，shape为[batch_size, sequence_length, hidden_size]。
        # encoder_attention_mask: 编码器的注意力掩码，用于进行跨层Attention，shape为[batch_size, 1, 1, sequence_length]。
        # past_key_values: 用于存储过去的Attention的键值对，可以用于增量计算，shape为[2, batch_size, num_heads, past_sequence_length, head_dim]。
        # use_cache: 是否使用过去的Attention的键值对进行增量计算。
        # output_attentions: 是否输出注意力权重。
        # output_hidden_states: 是否输出隐藏状态。
        # return_dict: 是否以字典形式返回结果。
        # bbox: 布局信息，用于计算空间Attention，shape为[batch_size, sequence_length, 4]，其中4表示边框坐标的数量。
        # position_ids: 位置信息，用于计算相对位置编码，shape为[batch_size, sequence_length]。
        # 该方法的具体实现需要根据具体模型的架构来确定，可以包括嵌套的神经网络层、卷积操作、Attention操作等等。
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 这里是初始化一些变量，这些变量用于存储模型的不同部分的输出。
        # 如果output_hidden_states为True，则all_hidden_states为一个空元组，用于存储所有隐藏状态的输出。
        # 如果output_attentions为True，则 all_self_attentions为一个空元组，用于存储所有自注意力层的输出；
        # all_cross_attentions也是一个空元组，如果add_cross_attention设置为True的话，则用于存储所有跨注意力层的输出。如果以上条件都不满足，则对应变量为None。
        next_decoder_cache = () if use_cache else None
        # 如果use_cache是False，则next_decoder_cache被设置为一个空元组，否则被设置为None。
        # 通常，缓存用于在预测期间重用计算结果，以减少计算量和提高效率。
        # 在这种情况下，如果不需要缓存，那么它就被设置为空元组，因为在后面的代码中可能会需要对next_decoder_cache执行追加操作。
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None
        # 这里的代码逻辑是用来计算相对位置编码，以在模型中添加注意力偏置。
        # 如果模型被配置为具有相对注意力偏置，则使用_cal_1d_pos_emb函数计算输入序列中不同位置之间的相对位置编码。
        # 同样，如果模型被配置为具有空间注意力偏置，则使用_cal_2d_pos_emb函数计算每个输入token的空间位置与其他输入token之间的相对位置编码。
        # 最终，函数返回这些相对位置编码（如果可用），供后续的自注意力和交叉注意力层使用。
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            #将当前的hidden_states添加到all_hidden_states元组中。这样，在整个循环结束后，all_hidden_states就会包含每个layer输出的hidden_states，可以用于后续分析和可视化。

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # 是用于提取当前层的头部掩码和过去的键值对（用于缓存）的。
            # head_mask和past_key_values是在模型的前向传递过程中传递给当前方法的输入参数，用于控制哪些头部参与计算，以及在使用过去的键值对时提供缓存。
            # 如果这些参数为None，则当前层将使用默认的头部掩码和不使用缓存的键值对。
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                # 是在检查模型是否启用了梯度检查点技术，该技术可以用来减少内存占用和计算量，特别适用于超大规模的模型。
                # 如果模型启用了该技术并且正在训练，代码会继续执行下去。
                # 如果没有启用该技术或者模型处于评估模式，则该分支代码不会被执行。
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False
                # 在检查在训练过程中是否开启了梯度检查点（gradient checkpointing）并且同时设置了使用缓存。
                # 如果开启了梯度检查点，使用缓存会导致错误，因此会将use_cache设置为False。
                # 梯度检查点是一种优化方法，可以减少在反向传播时需要保存的中间状态，以节省显存并加速训练。
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                # 这是定义一个create_custom_forward函数，用于创建一个自定义的前向传递函数。
                # 这个自定义的前向传递函数将past_key_value和output_attentions作为额外的参数传递给module。
                # 这个函数主要是为了在启用梯度检查点技术时定制前向传递。
                # 梯度检查点技术可以减少显存占用，但需要在前向传递过程中多次调用模型，因此需要自定义前向传递函数。
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
                # 是使用PyTorch中的checkpoint函数来实现梯度检查点技术，以减少计算图中的内存消耗。
                # 该函数的参数是一个自定义的前向传递函数create_custom_forward，它将传递给layer_module的输入作为其自己的输入，并在模块中使用传递的past_key_value和output_attentions变量。
                # 此外，还传递了其他参数hidden_states、attention_mask、layer_head_mask、encoder_hidden_states、encoder_attention_mask、rel_pos和rel_2d_pos。
                # 该函数返回layer_outputs，包括该层的输出结果和其他中间状态，如注意力矩阵和激活值。
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
                # 这是Transformer的一个前向传播计算，用于对输入进行处理并生成输出。其中，layer 是一个Transformer层的列表，每个元素都是一个Transformer层的实例。
                # hidden_states是输入的张量序列，attention_mask 是遮盖掩码，用于标记哪些位置需要被遮盖掉，layer_head_mask是每个头部的遮盖掩码，encoder_hidden_states是编码器的输出，
                # encoder_attention_mask是编码器遮盖掩码。
                # past_key_value是过去的键值对，output_attentions是是否返回注意力矩阵，rel_pos和rel_2d_pos是用于相对位置编码的矩阵。
                # 这个函数的输出是layer_outputs，其中包含该层的输出以及注意力矩阵等信息，具体内容取决于函数参数中output_hidden_states和output_attentions的取值。

            hidden_states = layer_outputs[0]
            #意思是将layer_outputs的第一个元素赋值给hidden_states，因为layer_outputs的第一个元素是每个编码层的输出（例如：每个位置的隐藏状态向量），所以这行代码的作用是更新模型当前的隐藏状态向量。
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            # 这段代码主要是将模型的输出结果存储到对应的变量中，用于后续的处理。
            # 其中，如果使用缓存，则将当前层的缓存状态添加到next_decoder_cache中，
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_self_attentions或者all_cross_attentions中，具体添加到哪个变量取决于模型配置是否使用了交叉注意力。

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 这段代码是用于将每一层的输出hidden_states存储到all_hidden_states中，以便之后返回所有层的输出。
        # 这是在模型进行前向传递时，如果设置了output_hidden_states参数为True，模型就会记录下每一层的输出。
        # 如果没有设置，all_hidden_states就会被设置为None。
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 这段代码是在判断是否需要以字典形式返回模型的输出结果。
        # 如果return_dict = True，则返回一个字典，包含模型输出结果的各个部分，如last_hidden_state、past_key_value、hidden_states、attentions和cross_attentions等。
        # 如果return_dict = False，则以元组的形式返回模型的输出结果，这个元组包含模型输出结果的各个部分，但是这些结果会根据是否为None进行过滤。
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        # 这段代码是一个函数的返回值，返回的是一个包含多个模型输出的对象BaseModelOutputWithPastAndCrossAttentions，其中包括：
        # last_hidden_state: 模型最后一层的输出，也就是预测的结果。
        # past_key_values: 用于存储前面几个序列位置的注意力权重的缓存，供后面序列位置的计算使用，可以提高计算效率。
        # hidden_states: 所有层的隐藏状态，如果在模型构建时设置了
        # output_hidden_states = True，则每一层的隐藏状态都会保存在这里。
        # attentions: 所有自注意力层的注意力权重，如果在模型构建时设置了
        # output_attentions = True，则每一层的注意力权重都会保存在这里。
        # cross_attentions: 所有交叉注意力层的注意力权重，如果在模型构建时设置了
        # output_attentions = True
        # 且模型有交叉注意力层，则每一层的注意力权重都会保存在这里。


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    # 这段代码定义了一个名为LayoutLMv2PreTrainedModel的类，该类继承了PreTrainedModel类。
    # PreTrainedModel类是Hugging Face Transformers库中所有预训练模型的基类。
    # 这个LayoutLMv2PreTrainedModel类是针对LayoutLMv2模型的预训练模型。
    # 该类中的config_class变量是用来定义对应的配置文件类；
    # pretrained_model_archive_map变量是一个字典，用来映射预训练模型的名称到对应的下载链接；
    # base_model_prefix变量则是定义了模型的名称前缀。
    # 此外，还有一个_keys_to_ignore_on_load_missing变量，用于忽略在加载时缺失的键（例如，旧版的模型中可能缺少某些键），以避免出现错误。
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    # 这段代码定义了一个类LayoutLMv2PreTrainedModel，它是PreTrainedModel的子类，因此继承了该类中的一些方法。
    # LayoutLMv2PreTrainedModel类还定义了一些类变量和参数，其中：config_class指定了该模型的配置文件类。
    # pretrained_model_archive_map是一个字典，包含了所有可用的预训练模型及其对应的资源地址。
    # base_model_prefix指定了模型名称的前缀，用于在调用save_pretrained和from_pretrained方法时指定。
    # _keys_to_ignore_on_load_missing是一个列表，其中包含的是从预训练模型加载模型参数时忽略的键的模式列表。
    def _init_weights(self, module):
        # 这是一个预训练模型类中的方法，用于对模型的参数进行初始化。
        # 该方法接受一个module参数，表示需要进行初始化的模型层。
        # 在该方法中，会根据模型层的类型对其参数进行不同的初始化，例如全连接层的参数会使用正态分布进行初始化，卷积层的参数会使用Xavier初始化方法进行初始化。
        # 这样可以让模型的参数在训练过程中更容易地进行优化，并且提高模型的收敛速度和性能。
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutLMv2LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 这段代码是在对神经网络的不同类型的层进行权重初始化，主要是使用不同的方法对线性层和嵌入层进行初始化，并对LayerNorm层进行了特殊的初始化。
        # 其中，对于线性层和嵌入层，使用正态分布随机初始化权重，标准差为self.config.initializer_range；
        # 对于LayerNorm层，将其偏置项初始化为0，权重项初始化为1。

def my_convert_sync_batchnorm(module, process_group=None):
    # 这个函数是一个自定义的PyTorch转换函数，用于将模型中的所有标准的nn.SyncBatchNorm层替换为我们自己编写的my_sync_batchnorm层。
    # nn.SyncBatchNorm是PyTorch官方提供的一种在分布式训练中使用的Batch Normalization层，它可以确保在分布式训练的过程中，不同进程的Batch Normalization层所计算的均值和方差是相同的。
    # my_sync_batchnorm则是我们自己编写的一种类似的Batch Normalization层，但使用的是我们自己实现的同步机制，用于处理在分布式训练中由于数据划分不均衡而导致的计算结果不一致的问题。
    # 这个函数接受一个PyTorch模型或者模型的一个子模块作为输入，遍历这个模型或子模块的所有层，如果发现某一层是 nn.SyncBatchNorm类型的层，就将其替换为my_sync_batchnorm类型的层。
    # 如果传入了一个process_group参数，那么在替换层的同时，还会为my_sync_batchnorm层指定相应的分布式进程组。
    # same as `nn.modules.SyncBatchNorm.convert_sync_batchnorm` but allowing converting from `detectron2.layers.FrozenBatchNorm2d`
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
        # 这个代码段中，通过判断输入的module是否是PyTorch的BatchNorm层（torch.nn.modules.batchnorm._BatchNorm的子类），来确定是否需要进行SyncBatchNorm的转换操作。
        # 如果是BatchNorm层，就调用nn.modules.SyncBatchNorm.convert_sync_batchnorm函数来进行转换。
        # 其中，process_group是一个torch.distributed.ProcessGroup对象，用于指定同步的进程组，如果不指定则默认为None。
    module_output = module
    #意思是将module作为函数输入后，将其输出赋值给变量module_output，可以理解为对module的处理结果。
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        # 是针对使用了detectron2库中的FrozenBatchNorm2d层的情况，将其转换为torch库中的SyncBatchNorm层。
        # SyncBatchNorm层是nn.BatchNorm的多GPU版本，它保证了在多GPU环境下模型的训练稳定性，通过将不同GPU上的数据进行同步，使得每个GPU上的数据都反映了整个batch的信息，从而提高了模型的泛化性能。
        # module_output.weight = torch.nn.Parameter(module.weight)
        # module_output.bias = torch.nn.Parameter(module.bias)
        # module_output.running_mean = module.running_mean
        # module_output.running_var = module.running_var
        # module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
        # 这段代码的作用是将module的参数赋值给module_output，使得module_output成为一个新的SyncBatchNorm模块。
        # 具体来说：module_output.weight是新模块的可训练参数，其值与原来的module的权重一致。
        # module_output.bias是新模块的可训练参数，其值与原来的module的偏置一致。
        # module_output.running_mean是新模块的非训练参数，其值与原来的module的运行均值一致。
        # module_output.running_var是新模块的非训练参数，其值与原来的module的运行方差一致。
        # module_output.num_batches_tracked是新模块的非训练参数，其值初始化为0，用于记录已经处理了多少个minibatch。
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
        # 这是一个递归函数，它遍历输入的module中的所有子模块，如果子模块是BatchNorm，则将其转换为SyncBatchNorm，然后将新模块加入module_output中。
        # 这样，函数最终返回一个新的模块，其中包含所有BatchNorm被替换为SyncBatchNorm的子模块。
    del module
        #del module的作用是删除一个变量或者对象，释放其占用的内存。在这个函数中，它的作用是删除一个PyTorch模块，将其从内存中释放掉。
    return module_output


class VisualBackbone(nn.Module):
    #这是一个PyTorch的nn.Module子类，用于实现视觉模型的主干(backbone)，通常用于计算机视觉任务中的图像特征提取。在这个类中，可能会包含一些卷积层、池化层、规范化层、残差连接等模型组件。
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        add_layoutlmv2_config(self.cfg)
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        # 这段代码定义了一个名为VisualBackbone的PyTorch模型类，继承自nn.Module。
        # 在初始化函数__init__中，它首先使用Detectron2库中的get_cfg函数获取配置文件，然后通过add_layoutlmv2_config函数将LayoutLMv2模型的配置信息添加到self.cfg中。
        # 接下来，它使用META_ARCH_REGISTRY字典从配置文件中获取模型的元模型名称，并实例化该元模型对象。
        # 然后，它断言该元模型的backbone属性是Detectron2库中的FPN对象，最后将该backbone对象保存到self.backbone属性中。
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            # 这段代码主要是用于判断是否需要将BatchNorm层转换为SyncBatchNorm层。
            # 当convert_sync_batchnorm为True，且当前环境满足分布式训练条件（即PyTorch的分布式模块已经初始化，且当前进程在分布式环境中），
            # 则将模型中的BatchNorm层转换为SyncBatchNorm层，以提高分布式训练的效率和精度。
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0
            # 这段代码的作用是在分布式训练中检查当前进程的GPU数量和分布式训练中的进程总数是否匹配，确保分布式训练的正确性。
            # 具体来说，首先获取当前进程的rank和node_size，然后获取总的进程数world_size，检查world_size是否是node_size的整数倍。
            # 如果不是，则抛出一个异常。
            # 这是因为在分布式训练中，多个进程会共享同一个模型，每个进程只会使用其中的一部分，而node_size代表的就是每个节点（node）上的GPU数量，world_size代表的是所有进程的总数。
            # 因此，确保world_size是node_size的整数倍可以确保每个进程都有对应的GPU可用，并且所有进程的GPU数量相同。
            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)
            ]
            # 这段代码创建一个嵌套的列表node_global_ranks，用于存储每个节点中所有GPU的全局排名（global rank）。
            # 具体来说，world_size是所有节点中GPU的总数，node_size是每个节点中GPU的数量，因此world_size // node_size表示节点的总数。
            # 然后，对于每个节点，使用range()函数生成从节点中第一个GPU的全局排名到最后一个GPU的全局排名的整数列表，并将这些列表存储在node_global_ranks中。
            # 例如，假设有两个节点，每个节点有两个GPU，那么world_size是4，node_size是2，node_global_ranks的值将是[[0, 1], [2, 3]]。
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
            ]
            # 这段代码是在将不同节点中的GPU分组，以便在每个组内实现BatchNormalization的同步更新。
            # 具体来说，node_global_ranks将不同节点的GPU编号分组，每个组包含node_size个GPU；sync_bn_groups则将每个组的GPU编号传入new_group函数中，
            # 创建出同步更新BatchNormalization时需要用到的通信组sync_bn_groups。
            # 每个通信组sync_bn_groups中包含的是同一节点内的多个GPU编号，这些GPU之间可以直接通信，因此同一组内的BatchNormalization可以直接进行同步更新。
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]
            # 这部分代码主要是对使用了Sync Batch Normalization进行分布式训练时的处理。
            # 首先检查是否需要使用Sync Batch Normalization，即需要进行分布式训练且使用了 Sync Batch Normalization，然后获取当前进程的rank，计算出节点的大小和全局大小，并确保全局大小是节点大小的倍数。
            # 接下来，将所有进程分成若干个节点，并对每个节点创建一个分组。然后，将当前进程分配到其所在的节点中，并确保其rank在节点的全局ranks中。
            self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        # 是在检查self.cfg.MODEL.PIXEL_MEAN和self.cfg.MODEL.PIXEL_STD是否具有相同的长度，因为它们代表图像像素的均值和标准差，需要确保它们在每个通道上都有相应的值。
        # 然后，num_channels将被分配为像素均值和标准差的长度。
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        # 将self.cfg.MODEL.PIXEL_MEAN的值转换为一个张量，并将其注册为一个buffer。这个buffer的名字是pixel_mean。
        # 这里使用register_buffer()方法将这个buffer注册为模型的buffer，这样它就会在反向传播过程中自动被跟踪和更新。
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))
        self.out_feature_key = "p2"
        # 是在VisualBackbone类的构造函数中，将pixel_mean和pixel_std注册为buffer。buffer是与model parameters类似的状态变量，但是不会被autograd追踪，也不会被更新。
        # 在模型保存和加载时，buffer也会被保存和加载。
        # pixel_mean和pixel_std是用来对输入图像进行归一化的，将输入的像素值减去均值再除以标准差，可以使得数据更易于训练。
        # self.out_feature_key则是指定输出的特征图的名称，这里指定为p2。
        if torch.is_deterministic():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
            # 这段代码实现了一个池化层，根据输入特征图的大小和后续计算需要的输出特征图大小进行自适应池化或平均池化。
            # 如果是确定性计算模式，则使用平均池化代替自适应池化。
            # 具体来说，如果是自适应池化，它将输入特征图大小转换为输出特征图大小，这个大小由输入参数config.image_feature_pool_shape指定，这个参数一般设置为(1, 1)，所以实际上是将特征图压缩到一个点。
            # 如果是平均池化，它将输入特征图大小按照backbone_stride转换为输出特征图大小。
            # 最终的池化方式是nn.AvgPool2d。
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
            # 这段代码创建了一个AdaptiveAvgPool2d层，用于在可变输入大小的情况下将输入池化为一个固定大小的特征图。
            # AdaptiveAvgPool2d会自动计算输出大小，并在每个空间维度上应用一个平均池化层。
            # 具体来说，这里使用了配置文件中给定的image_feature_pool_shape参数来指定输出特征图的大小。
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
            #这段代码的作用是检查config.image_feature_pool_shape的长度是否为2，如果是，则将backbone输出的通道数添加到其中，以便在最终的线性层中使用。该操作的目的是根据配置文件动态地调整最终特征向量的维度。
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]
        #这个assertion语句是用来检查特征提取器的输出通道数是否与配置文件中设置的图像特征池化形状中的通道数一致。如果不一致，说明存在配置错误，抛出AssertionError异常。

    def forward(self, images):      
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features
        # 这是一个PyTorch的forward函数，它接收一个images输入，执行以下操作：
        # 将输入images转换为tensor形式，如果images已经是tensor则不需要转换。
        # 根据模型的配置对输入进行归一化，使用预定义的mean和std。
        # 将输入送入backbone提取图像特征。
        # 获取backbone输出中的指定特征（self.out_feature_key）。
        # 对特征执行自适应平均池化（如果是非确定性训练，则使用AvgPool2d代替AdaptiveAvgPool2d），并将特征展平。
        # 返回处理后的特征。


class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
        # 这段代码是Python语言中的类定义，定义了一个名为LayoutLMv2Model的类，它是从LayoutLMv2PreTrainedModel类继承而来的。
        # 在深度学习中，通常使用类来定义模型结构，而这个类的定义包含了LayoutLMv2Model类的名称，以及它继承自LayoutLMv2PreTrainedModel类的属性和方法。
        # 具体来说，LayoutLMv2Model类可以通过调用其父类LayoutLMv2PreTrainedModel的方法和属性，来实现基于预训练模型的文本和布局分析任务。
        # 这里的代码片段可能是使用Hugging Face Transformers库创建一个名为LayoutLMv2Model的文本和布局分析模型的示例。
    def __init__(self, config):
        super(LayoutLMv2Model, self).__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)
        # 这段代码是LayoutLMv2Model类的构造函数，用于初始化类的实例。
        # 具体来说：__init__方法是Python中的特殊方法，用于创建类的实例。
        # 在这个方法中，使用super(LayoutLMv2Model, self).__init__(config)语句调用了父类LayoutLMv2PreTrainedModel的构造函数，以确保父类中的属性和方法被正确地初始化。
        # self.config = config将输入的config对象保存为类的属性，以便其他方法可以访问。
        # self.has_visual_segment_embedding = config.has_visual_segment_embedding从config对象中获取has_visual_segment_embedding属性的值，并将其保存为类的属性。
        # self.embeddings = LayoutLMv2Embeddings(config)创建了一个名为self.embeddings的属性，并使用config对象作为输入，创建了一个新的LayoutLMv2Embeddings类的实例，并将其赋值给该属性。
        # LayoutLMv2Embeddings类是用于将输入数据转换为模型输入格式的一个重要组件。
        # 综上，这段代码是在LayoutLMv2Model类的构造函数中对类的属性进行初始化，其中包括config对象、has_visual_segment_embedding属性和embeddings属性。
        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        # 这段代码是LayoutLMv2Model类的构造函数的一部分，用于创建类的实例属性。
        # self.visual = VisualBackbone(config)创建了一个名为self.visual的属性，并使用config对象作为输入，创建了一个新的VisualBackbone类的实例，并将其赋值给该属性。
        # VisualBackbone是用于从输入的视觉特征提取器中提取视觉特征的组件，通常用于处理与文本相关的视觉信息。
        # self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)创建了一个名为self.visual_proj的属性，
        # 并使用config对象中的image_feature_pool_shape属性和hidden_size属性作为输入，创建了一个新的线性层(nn.Linear)的实例，并将其赋值给该属性。
        # 该线性层的输入维度为config.image_feature_pool_shape[-1]，即视觉特征的维度，输出维度为config.hidden_size，即模型的隐藏状态维度。
        # 这个线性层通常用于将从视觉特征提取器中提取出的视觉特征映射到模型的隐藏状态空间中，以便与文本信息进行融合。
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
            # 这段代码是LayoutLMv2Model类的构造函数的一部分，用于创建类的实例属性。
            # self.has_visual_segment_embedding是在config对象中定义的一个布尔属性，用于表示模型是否包含视觉分割嵌入(visual segment embedding)。
            # if self.has_visual_segment_embedding: 表示如果模型包含视觉分割嵌入，则执行下面的代码块。
            # nn.Embedding(1, config.hidden_size).weight[0]创建了一个大小为(1, hidden_size)的嵌入矩阵，并返回其第一行，即一个大小为(hidden_size, )的张量。
            # 这个嵌入矩阵可以用来对视觉区域进行编码，以在模型中引入位置信息。nn.Parameter()将一个张量封装为模型参数，并自动跟踪其梯度。
            # self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])创建了一个名为self.visual_segment_embedding的模型参数，并将上面创建的张量作为其值。
            # 这个模型参数通常用于表示视觉区域的位置信息，可以在模型的前向传播中与其他特征进行拼接。
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)
        # 这段代码是 LayoutLMv2Model类的构造函数的一部分，用于创建类的实例属性。
        # self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)创建了一个名为self.visual_LayerNorm的属性，
        # 并使用 config对象中的hidden_size和layer_norm_eps属性作为输入，创建了一个新的LayerNorm层的实例，并将其赋值给该属性。
        # LayerNorm是一种归一化方法，用于标准化输入特征的均值和方差，以使其具有零均值和单位方差。这个层通常用于处理来自视觉特征提取器的视觉特征，以便它们与来自文本嵌入的特征具有相同的分布。
        # self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)创建了一个名为self.visual_dropout的属性，并使用config对象中的hidden_dropout_prob属性作为输入，
        # 创建了一个新的Dropout层的实例，并将其赋值给该属性。
        # Dropout是一种正则化方法，用于随机地丢弃网络中的一些神经元，以减少过拟合。这个层通常用于处理来自视觉特征提取器的视觉特征，以防止模型过拟合训练数据。
        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)
        # 这段代码是LayoutLMv2Model类的构造函数的一部分，用于创建类的实例属性。
        # self.encoder = LayoutLMv2Encoder(config)创建了一个名为self.encoder的属性，并使用config对象作为输入，创建了一个新的LayoutLMv2Encoder层的实例，并将其赋值给该属性。
        # LayoutLMv2Encoder是多层双向Transformer编码器的堆叠，用于对输入的文本序列和位置嵌入进行编码。
        # 它是LayoutLMv2Model的核心组件之一，用于实现文本和视觉特征的联合编码。
        # self.pooler = LayoutLMv2Pooler(config)创建了一个名为self.pooler的属性，并使用config对象作为输入，创建了一个新的LayoutLMv2Pooler层的实例，并将其赋值给该属性。
        # LayoutLMv2Pooler是用于生成池化特征的组件，将编码器的最后一层隐藏状态的均值作为序列级别表示，以用于序列分类任务。
        # 这个池化特征通常用于将编码器的输出转换为固定长度的向量，以便将其输入到一个全连接层或分类器中。
        self.init_weights()
        # 它是LayoutLMv2Model类中的一个方法，用于对模型的权重进行初始化。在这个方法中，模型的各个组件的权重都会被初始化为随机数，以确保模型在训练时具有足够的初始随机性，从而帮助模型更好地学习输入数据的特征。
        # 通常，权重初始化的目标是将它们初始化为一个合适的范围，以便它们在初始状态下具有相对均匀的分布。这可以避免在训练期间出现梯度消失或爆炸的问题。
        # 在LayoutLMv2Model类中，init_weights方法会根据config对象中的参数，使用PyTorch内置的normal_和uniform_方法来对各个组件的权重进行初始化。
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        # get_input_embeddings是LayoutLMv2Model类中的一个方法，用于获取输入嵌入层，即获取文本序列中每个词的嵌入表示。在这个方法中，它返回了self.embeddings.word_embeddings，
        # 即LayoutLMv2Embeddings中的词嵌入层。
        # 该层接受一个包含文本序列中每个词对应的ID的整数张量作为输入，并将每个词的ID转换为对应的嵌入表示。
        # 最终，它返回一个与输入文本序列相同大小的嵌入张量。
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        # set_input_embeddings是LayoutLMv2Model类中的一个方法，用于设置输入嵌入层的权重。在这个方法中，它将value参数设置为self.embeddings.word_embeddings，即LayoutLMv2Embeddings中的词嵌入层的权重。
        # 通常情况下，我们可以使用已经训练好的嵌入层来初始化一个新的模型，以便该模型能够使用该预训练嵌入层来对输入文本序列进行编码。
        # 在某些情况下，我们可能想要在训练过程中对输入嵌入层的权重进行微调或重新训练，因此可以使用set_input_embeddings方法来设置输入嵌入层的权重。
        # 例如，在迁移学习中，我们可以将一个预先训练好的模型的输入嵌入层权重初始化到一个新模型中，并将其作为该模型的初始权重，然后在新任务的训练中微调该权重。
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
        # _prune_heads是LayoutLMv2Model类中的一个方法，用于剪枝模型中的多头注意力层的头。
        # 该方法接收一个包含要剪枝的多头注意力层头的字典heads_to_prune，该字典的键表示要剪枝的层，而值则是一个包含要剪枝的头的列表。
        # 在该方法内部，首先通过heads_to_prune中的字典来确定需要进行剪枝的具体层以及需要剪枝的头。
        # 然后，该方法会对每个要剪枝的头调用相应层的多头注意力层的prune_heads方法来进行剪枝。
        # 在这个方法中，self.encoder.layer[layer].attention返回相应层的多头注意力层，而prune_heads方法则用于剪枝该多头注意力层的指定头。
        # 通过剪枝模型中的多头注意力层的头，我们可以减少模型的计算量，从而提高模型的速度和效率，同时还可以提高模型的泛化能力和鲁棒性。
    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        seq_length = input_ids.size(1)
        # _calc_text_embeddings是LayoutLMv2Model类中的一个方法，用于计算输入文本序列的嵌入向量。
        # 该方法接收输入文本序列的词ID(input_ids)、词框信息(bbox)、位置ID(position_ids)和标记类型ID(token_type_ids)等参数。
        # 在该方法内部，首先通过input_ids参数获取输入文本序列的长度seq_length，然后将input_ids、bbox、position_ids和token_type_ids传递给self.embeddings中的forward方法，
        # 该方法返回一个元组，其中包含输入文本序列的嵌入向量。
        # 具体而言，seq_length表示输入文本序列的长度，input_ids是一个形状为[batch_size, seq_length]的张量，
        # 其中每个元素都是一个词的ID，bbox是一个形状为[batch_size, seq_length, 4]的张量，
        # 其中每个元素都是一个包含左上角和右下角坐标的边界框，position_ids是一个形状为[batch_size, seq_length]的张量，
        # 其中每个元素都是对应词的位置ID，token_type_ids是一个形状为[batch_size, seq_length]的张量，其中每个元素都是对应词的标记类型ID。
        # 在LayoutLMv2模型中，输入文本序列的嵌入向量是通过将输入文本序列的词嵌入向量、位置嵌入向量和标记类型嵌入向量进行拼接得到的。
        # 具体而言，每个词的嵌入向量和位置嵌入向量都会与一个特定的边界框向量相乘，然后将它们与标记类型嵌入向量进行拼接。
        # 该方法返回一个元组，其中第一个元素是拼接后的嵌入向量，第二个元素是注意力掩码。
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            # 这段代码实现了为输入的input_ids序列生成位置编码position_ids的功能。
            # 如果传入的position_ids参数为空，则会根据输入序列的长度seq_length生成一个形状为(1, seq_length)的Tensor，其中包含了从0到seq_length - 1的整数序列。
            # 然后使用unsqueeze方法将这个Tensor的维度从(1, seq_length)转变为(batch_size, seq_length)，并使用expand_as方法将其扩展为和输入input_ids的形状相同。
            # 这样生成的position_ids序列将作为位置编码传递给模型的输入层，用于为不同位置的单词或子词在嵌入空间中建立独立的表示。
            # 位置编码是Transformer等神经网络中常用的一种技术，用于为输入序列中的不同位置引入位置信息，从而提高模型的表现能力。
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            # 如果token_type_ids为空，则将其初始化为与input_ids具有相同大小的全零张量。

        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox_x = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[1]
        )
        visual_bbox_y = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[0]
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand_as(input_ids)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv2ForRelationExtraction(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = REDecoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        labels=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        entities=None,
        relations=None,
    ):
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return ReOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )
