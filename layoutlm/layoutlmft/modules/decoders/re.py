import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BiaffineAttention(torch.nn.Module):
    # 这行代码定义了一个继承自PyTorch的torch.nn.Module类的新类BiaffineAttention。BiaffineAttention是一个用于实现双仿射注意力机制的神经网络模型。
    # 双仿射注意力机制是一种常用的注意力机制，用于在一个序列中对不同位置的信息进行加权。在自然语言处理中，双仿射注意力机制经常用于句子级别的任务，例如情感分析、命名实体识别和语言翻译等任务。
    # 具体而言，双仿射注意力机制通过计算输入向量的两个仿射变换的点积来得到一个矩阵，该矩阵表示每个位置与其他位置之间的关联度。然后，通过对每一行或每一列进行softmax操作，可以得到每个位置与其他位置的权重，用于进行加权。
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
        这是对一个双仿射注意力算子的PyTorch实现，可以用于二元关系分类任务的分类器。
        该算子需要输入两个具有相同特征维度的张量x_1和x_2，并且根据输入的特征维度大小定义了一个输入特征维度in_features和输出特征维度out_features。在应用该算子之后，会返回一个具有输出特征维度大小的张量。
        该实现是基于论文"End-to-end neural relation extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275)中描述的算子，该算子在二元关系分类任务中得到了较好的结果。
        代码中给出了一个简单的示例，其中定义了一个BiaffineAttention类的实例，输入两个随机初始化的张量x_1和x_2，并应用算子，最后打印输出的张量大小。
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    # 这段代码是一个类的初始化方法，其中包含了一些变量的定义和初始化以及一些子模块的定义和初始化。这个类被称为BiaffineAttention。
    # 具体来说，这个类有两个输入参数in_features和out_features，分别表示输入特征和输出特征的维度。在初始化方法中，这两个参数被保存为类的成员变量self.in_features和self.out_features。
    # 这个类包含两个子模块：一个双线性模块（bilinear）和一个线性模块（linear）。双线性模块接受两个输入，并输出一个out_features维度的向量；线性模块接受两个in_features维度的向量拼接在一起的输入，并输出一个out_features维度的向量。
    # 在初始化方法中，这两个子模块都被创建，并保存为类的成员变量。此外，还调用了一个方法reset_parameters()，用于初始化模型的参数。这个方法的具体实现没有给出，但是一般是用于随机初始化模型参数的。

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

        # 这是一个PyTorch模型中的forward方法和reset_parameters方法。
        # forward方法是神经网络模型的前向传递计算过程，接收两个输入张量x_1和x_2，其中x_1和x_2具有相同的形状，并将它们分别输入到self.bilinear和self.linear中，得到两个输出张量，分别是双线性映射和线性映射。
        # 接着，将这两个输出张量相加作为最终的输出。
        # reset_parameters方法是一个函数，用于对神经网络模型中的参数进行重新初始化。
        # 在该函数中，调用self.bilinear和self.linear中的reset_parameters方法，将它们的参数重新初始化为随机值，以便在训练过程中重新优化这些参数。


class REDecoder(nn.Module):
    # 这是一个PyTorch的神经网络模型类REDecoder，用于实现关系抽取（Relation Extraction）任务中的解码器（Decoder）部分。
    # 关系抽取是自然语言处理中的一个重要任务，目标是从给定的文本中抽取出描述实体之间关系的文本片段。在这个任务中，解码器通常用于将经过编码器（Encoder）处理过的文本特征进行解码，得到对实体之间关系的分类结果。
    # REDecoder类的具体实现可能因应用场景而有所不同，但通常包含多层神经网络结构，例如全连接层、循环神经网络、卷积神经网络等。它的输入通常是编码器的输出和实体的特征表示，输出则是针对给定实体对的预测结果。
    # 需要注意的是，代码中的nn表示PyTorch中的神经网络模块，是PyTorch的一个子模块。
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()
        # 这是一个PyTorch的神经网络模型类REDecoder的初始化方法__init__()，用于定义模型的各个组件。
        # 在该方法中，首先调用super().__init__()初始化nn.Module的父类，然后定义了一个名为entity_emb的嵌入层（Embedding Layer），用于将实体的类别（共有三类）映射到对应的隐层向量表示，
        # 其中config.hidden_size表示隐层向量的维度。
        # 接着，定义了一个名为projection的序列模块（Sequential Module），包含两个全连接层、两个ReLU激活函数、两个Dropout层，
        # 最后一个全连接层的输出维度为config.hidden_size // 2，即隐层向量维度的一半。这个projection模块会被用于处理经过编码器得到的实体表示和文本表示，用于预测实体对之间的关系。
        # 然后，分别对头实体和尾实体应用深拷贝（deepcopy）得到两个名为ffnn_head和ffnn_tail的完全相同的前向传递网络（feedforward neural network），用于分别处理头实体和尾实体的隐层向量表示。
        # 最后，定义一个名为rel_classifier的双仿射注意力模块（Biaffine Attention），用于将头实体和尾实体的隐层向量表示结合起来，并预测它们之间的关系。config.hidden_size // 2
        # 表示每个实体的隐层向量维度的一半，输出维度为2，用于表示关系是否存在。同时定义一个名为loss_fct的交叉熵损失函数（CrossEntropyLoss），用于计算模型预测值和真实值之间的损失。

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        # 这是一个Python函数build_relation，它的作用是在entities列表中构建每个实体对应的关系矩阵，返回一个新的关系矩阵列表new_relations。
        # 输入参数relations是一个batch_size大小的列表，其中每个元素是一个形状为(num_relations, 2)的张量，表示当前批次中每个样本对应的实体对以及它们之间的关系。
        # entities是一个batch_size大小的列表，其中每个元素是一个形状为(num_entities, hidden_size)的张量，表示当前批次中每个样本对应的实体向量表示。
        # 在函数中，首先获取batch_size的值，然后定义一个空列表new_relations用于存放新的关系矩阵。接下来，通过for循环遍历当前批次中的每个样本。
        # 对于每个样本，首先获取实体的数量num_entities，然后使用torch.zeros()函数创建一个形状为(num_entities, num_entities, 2)的零张量relation_matrix，其中2表示关系矩阵的维度。
        # 接着，使用for循环遍历当前样本中的每个实体对(i, j)，并将关系矩阵中对应的(i, j)和(j, i)位置的值分别设置为该实体对的关系标签（0或1），这里的关系标签是根据relations参数中的实体对关系获取的。
        # 最后，将新构建的关系矩阵添加到new_relations列表中，并在所有样本处理完成后返回该列表。
        # 总的来说，这个函数的作用是将实体向量表示和实体对关系信息整合起来，构建出每个实体对应的关系矩阵。
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            # 这段代码是一个函数build_relation的实现。该函数将一个批次的关系和实体标签数据作为输入，并返回一个新的关系列表，其中包含所有可能的实体对之间的关系。
            # 如果一个批次中只有少于等于2个实体，那么函数会将它们视为同一个实体并将其标签设置为零。
            # 函数首先遍历每个样本（batch），然后检查该样本中实体的数量。如果实体数小于等于2，则该样本将被视为同一个实体，并且将在下一步中将其标签设置为零。
            # 如果实体数大于2，则该函数将创建一个集合all_possible_relations，其中包含了所有实体对之间的关系，其中起始实体标签为1（表示主体）且终止实体标签为2（表示客体）。
            # 例如，如果实体标签为[1, 0, 2]，则该函数将返回一个元组(0, 2)，其中0表示主体的位置，2表示客体的位置。
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                    len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
            # 这段代码主要是对于一个名为relations的列表中的元素进行处理，并生成一个新的列表new_relations。
            # 下面是具体的处理过程： 如果all_possible_relations为空，则将(0, 1)添加到all_possible_relations集合中。
            # 从relations[b]["head"]和relations[b]["tail"]中提取元素，生成一个元组列表positive_relations。
            # 生成另一个元组列表negative_relations，其中包含all_possible_relations中的元素，但不在positive_relations中。
            # 将positive_relations中的元组与all_possible_relations中的元组进行比较，保留两个集合中都存在的元组，生成一个新的列表reordered_relations。
            # 创建一个新的字典relation_per_doc，其中包含键head，tail和label。将reordered_relations中的头和尾分别添加到relation_per_doc["head"]和relation_per_doc["tail"]列表中。
            # 创建一个label列表，其中包含1的数量等于positive_relations中的元素数量，其余元素的值为0，将其添加到relation_per_doc["label"]列表中。
            # 断言relation_per_doc["head"]不为空。
            # 将relation_per_doc添加到new_relations列表中。
            # 综上所述，这段代码是将一个包含关系数据的列表进行处理，生成一个新的列表，其中每个元素表示一个文档中的关系及其标签。
            # 其中，positive_relations是已知的关系，negative_relations是可能的但未知的关系，relation_per_doc包含一个文档的所有关系以及它们的标签。

        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations
        # 这段代码定义了一个函数get_predicted_relations，它的作用是从模型的预测结果中获取关系的预测结果。函数接受三个参数：
        # logits：模型输出的关系分类得分矩阵，shape为(batch_size, num_relations, 2)，表示每个关系是否存在的概率，其中第二维是关系数量，第三维是二分类的得分。
        # relations：输入模型的关系信息，其中包括每个关系的头尾实体在文本中的位置。
        # entities：输入模型的实体信息，其中包括每个实体在文本中的起始和结束位置以及实体类型。
        # 在函数中，首先遍历logits的第一维，判断是否为关系1（预测存在关系），如果不是则跳过该关系。
        # 对于预测存在的关系，从relations和entities中获取头实体和尾实体的位置信息、类型信息，并将预测结果添加到pred_relations列表中。
        # 最后函数返回pred_relations列表，其中包含了所有预测存在的关系。

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        # 这是一个神经网络的前向传递函数，用于计算模型的预测值和损失。下面是该函数的具体实现：
        # 获取输入的hidden_states张量的形状(batch_size, max_n_words,context_dim)，其中batch_size表示当前批次的大小，max_n_words表示每个文档中的最大单词数，context_dim表示上下文嵌入的维度。
        # 获取hidden_states张量所在的设备类型。调用build_relation方法，将输入的entities和relations转换为模型所需的格式。
        # 初始化损失loss为0。初始化一个空的列表all_pred_relations，用于存储所有文档的预测关系。
        # 对于每个文档：提取当前文档中的实体嵌入和关系嵌入。
        # 将实体嵌入和关系嵌入进行拼接，得到一个形状为(num_relations, 2 * entity_dim)的张量，其中num_relations表示当前文档中的关系数量，entity_dim表示实体嵌入的维度。
        # 将该张量传入全连接层，得到一个形状为(num_relations, num_classes)的输出张量，其中num_classes表示模型的输出类别数。
        # 对输出张量进行sigmoid操作，得到每个预测关系的概率。将预测关系添加到all_pred_relations列表中。
        # 根据relation_per_doc中的标签，计算交叉熵损失，并将其加入到总损失中。返回all_pred_relations列表和总损失。
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)
                # 这段代码是将实体和关系的索引和标签信息从数据字典中提取出来，并进行处理。
                # 首先，它从输入数据中取出关系的头实体、尾实体和关系标签，以及实体的起始位置和标签信息，并将它们转换成PyTorch中的Tensor数据类型，并将它们放到指定的设备（比如GPU）上进行计算。
                # 然后，它通过头实体的索引信息，找到对应的实体在实体序列中的起始位置，并从实体标签信息中取出头实体的标签。
                # 接着，它将头实体的标签通过一个实体嵌入层（Entity Embedding）映射成一个实体表示（Entity Representation），这个表示用于后续的特征提取和关系预测。
                # 可以看到，这段代码是将输入数据中的实体和关系信息进行了处理和编码，为后续的模型计算和预测做好了准备。
            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
                # 这段代码是将头实体和尾实体的特征向量计算出来，并将它们拼接成一个更综合的表示。
                # 首先，它通过尾实体的索引信息，找到对应的实体在实体序列中的起始位置，并从实体标签信息中取出尾实体的标签。
                # 接着，它将尾实体的标签通过一个实体嵌入层（EntityEmbedding）映射成一个实体表示（Entity Representation），这个表示同样用于后续的特征提取和关系预测。
                # 然后，它从模型的输出中取出与头实体和尾实体对应的特征向量，并将它们和对应的实体标签表示拼接起来，得到一个更综合的实体 - 标签表示。
                # 这个拼接操作使用 PyTorch中的 torch.cat函数实现，它将两个张量按照指定的维度进行拼接。
                # 最终，这个综合表示将作为输入数据，经过两个前馈神经网络（FFNN）的映射，得到头实体和尾实体的预测向量，以便后续的关系预测。
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
                # 这段代码是对经过处理的实体 - 标签表示进行前向传播，得到关系的预测结果，并计算损失。
                # 首先，它将经过拼接后的头实体和尾实体的表示，分别输入两个前馈神经网络（FFNN）中进行映射，得到头实体和尾实体的预测向量。这个操作可以理解为将实体和实体标签的信息通过神经网络映射成为一个维度更低的向量。
                # 接着，它将头实体和尾实体的预测向量作为输入，通过关系分类器（Relation Classifier）进行关系预测。
                # 这个操作可以理解为将头实体和尾实体之间的关系通过神经网络映射成为一个预测值。
                # 然后，它计算模型的损失值（Loss），将预测结果与真实的关系标签之间的误差作为模型的损失值。这个操作可以理解为模型在训练过程中根据预测结果的准确程度对自身进行调整，以使得预测结果更加准确。
                # 接着，它通过get_predicted_relations函数，将预测结果转换为实际的关系类型，并将其添加到all_pred_relations列表中。
                # 总的来说，这段代码是将处理后的实体和关系信息输入到神经网络中进行前向传播，并得到模型的预测结果和损失值，以便后续的模型调整和评估。
        return loss, all_pred_relations
