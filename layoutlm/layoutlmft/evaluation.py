import os
import re

import numpy as np

from transformers.utils import logging


logger = logging.get_logger(__name__)


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    # 这段代码定义了一个正则表达式对象_re_checkpoint，用于匹配以checkpoint-开头，后跟数字的字符串。具体来说，它通过re.compile()函数将一个正则表达式字符串编译为一个正则表达式对象。正则表达式字符串的含义如下：
    # ^ 表示匹配字符串的开头
    # PREFIX_CHECKPOINT_DIR 是一个字符串变量，表示匹配的字符串必须以该变量的值开头，即checkpoint
    # \-(\d+) 表示匹配一个短横线，后面跟着一个或多个数字，并将数字部分作为一个捕获组。其中\-表示匹配一个短横线，\d+表示匹配一个或多个数字。
    # $ 表示匹配字符串的结尾。
    # 这个正则表达式的作用是检查一个字符串是否符合形如checkpoint-数字的格式，如果符合，那么可以从字符串中提取出数字部分。
    # 在这段代码中，该正则表达式对象将用于解析模型训练过程中的检查点文件名。

def get_last_checkpoint(folder): 
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))
        # 这段代码定义了一个函数get_last_checkpoint(folder)，它的作用是从指定的文件夹中获取最新的检查点文件路径。函数的具体行为如下：
        # 首先，它列出指定文件夹中的所有文件和子文件夹，并将它们的名称存储在一个列表content中。
        # 然后，它通过列表推导式生成一个名为checkpoints的新列表。这个列表包含了所有名称符合检查点文件格式（即以checkpoint-数字的格式命名）且是文件夹的路径。
        # 如果checkpoints列表为空，那么函数将返回None。
        # 否则，函数将返回checkpoints列表中最新的检查点文件路径。它通过调用max()函数和一个key参数来实现这个功能。key参数是一个函数，用于将列表中的每个元素映射到一个数字。在这个函数中，它使用正则表达式对象_re_checkpoint来解析文件名，并从中提取出数字部分，然后将该数字转换为整数并返回。最后，max()函数会返回一个具有最大值的元素，即最新的检查点文件路径。
        # 总之，这个函数的作用是查找指定文件夹中最新的检查点文件，并返回其路径。

def re_score(pred_relations, gt_relations, mode="strict"):
        # 这是一个Python函数的定义，接受三个参数：pred_relations，gt_relations和mode，并返回一个结果。
        # 该函数的功能可能是用于评估自然语言处理任务中预测的关系（relations）和标准答案之间的匹配度。其中：
        # pred_relations是一个列表或数组，表示模型预测的关系；
        # gt_relations是一个列表或数组，表示标准答案中的关系；
        # mode参数表示匹配模式，有可能是 "strict"（严格匹配）或"relaxed"（宽松匹配）。
        # 具体而言，这个函数会比较pred_relations和gt_relations中的元素，计算它们之间的匹配程度，并返回一个得分或评估指标，用于衡量模型在关系匹配方面的表现。
        # 不同的匹配模式可能会产生不同的得分。
    """Evaluate RE predictions

    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations

            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}

        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries'"""

    assert mode in ["strict", "boundaries"]
            # 这是一个Python代码中的语句，它用于检查一个名为"mode"的变量是否属于指定的列表["strict", "boundaries"]中的一个。
            # 如果变量的值不是这两个值中的一个，那么代码可能会引发一个异常或者进行一些其他的错误处理。
            # 换句话说，这行代码要求"mode"变量只能是"strict"或"boundaries"中的一个，如果"mode"变量的值不在这两个选项中，则代码将无法继续执行，可能会抛出一个异常或者进行其他的错误处理。
    relation_types = [v for v in [0, 1] if not v == 0]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
            # 这是一个Python代码中的语句，它定义了两个变量：relation_types和scores。以下是这两个变量的含义：
            # relation_types变量：这是一个列表推导式，它创建了一个包含所有非零值的列表。具体来说，它从列表[0, 1]中筛选出非零值，将它们存储在relation_types变量中。
            # scores变量：这是一个字典推导式，它创建了一个包含所有关系类型的字典，以及一个额外的"ALL"关系类型。
            # 具体来说，它使用relation_types变量中的关系类型作为字典的键，并为每个关系类型创建一个字典，其中包含三个键："tp"、"fp"和"fn"，分别表示true positives、false positives和false negatives的数量。
            # 最后，它将一个额外的"ALL"键添加到字典中，以便可以跟踪所有关系类型的总体性能。
            # 因此，这行代码的作用是定义一个包含指定关系类型及其性能统计的字典，并将其存储在变量scores中。
    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
            # 这段代码是使用Python中的zip()函数来对两个列表pred_relations和gt_relations进行逐一配对，对应位置上的元素组成一个元组，并返回一个由这些元组组成的迭代器。
            # 具体解释如下：pred_relations: 模型预测出来的关系列表，其中每个元素是一个包含多个关系的列表（如果该句子中有多个关系），每个关系用一个元组表示。
            # gt_relations: 标注数据中的关系列表，与pred_relations格式相同，每个元素也是一个包含多个关系的列表。
            # zip(pred_relations, gt_relations)将这两个列表逐一配对，取出每个位置上对应的两个列表中的子列表，将它们组成一个元组(pred_sent, gt_sent)，
            # 这个元组包含了一个模型预测的句子中的关系列表和标注数据中对应的关系列表。
            # 接下来，这个元组可以用于进行比较、计算指标等操作。通常在关系抽取任务中，可以用这个元组来计算预测结果和标注数据的精确度、召回率、F1值等指标。
        for rel_type in relation_types:
                # 这段代码是一个for 循环语句，relation_types 是一个列表，其中包含了所有可能的关系类型。
                # for rel_type in relation_types: 会依次遍历这个列表中的每一个元素，并将其赋值给变量
                # rel_type。在循环体中，可以使用rel_type变量来表示当前遍历到的关系类型，并执行相应的操作。
                # 通常在关系抽取任务中，需要将模型预测的结果与标注数据进行比较，判断预测出来的关系类型是否正确，因此可以用这个循环遍历所有可能的关系类型，依次计算模型的精确度、召回率、F1值等指标。
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {
                    (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                    for rel in pred_sent
                    if rel["type"] == rel_type
                }
                gt_rels = {
                    (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                    for rel in gt_sent
                    if rel["type"] == rel_type
                }
                # 这段代码是用于在关系抽取任务中计算模型预测结果与标注数据的精确度、召回率、F1值等指标，具体解释如下：
                # mode == "strict"：mode是一个字符串变量，用于指定计算指标时采用的模式，此处指定的是"strict"模式，表示对模型预测结果与标注数据的关系类型和实体位置都要进行严格匹配。
                # pred_rels：用于存储模型预测出来的某个关系类型rel_type的关系三元组，其中每个元素是一个由(head, head_type, tail, tail_type)组成的元组，表示一个关系实例，head和tail分别表示实体的位置，
                # head_type和tail_type表示实体类型。
                # gt_rels：用于存储标注数据中的某个关系类型rel_type的关系三元组，格式同pred_rels。
                # (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])：这行代码提取了一个关系元素rel中的实体位置和类型信息，并将其组成一个关系三元组，
                # 存储到pred_rels或gt_rels中。具体来说，rel["head"]表示关系中头实体的位置，rel["head_type"]表示头实体的类型，rel["tail"]和rel["tail_type"]分别表示关系中尾实体的位置和类型。
                # if rel["type"] == rel_type：这个条件语句用于判断当前遍历到的关系元素 rel 是否属于指定的关系类型 rel_type，如果是，则将其加入到 pred_rels 或 gt_rels 中。
            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}
                # 这段代码与上一段代码类似，用于计算模型预测结果与标注数据的精确度、召回率、F1值等指标，但是不同的是，这里的模式mode是 "boundaries"，表示只要求模型预测的实体位置与标注数据中实体位置相交即可。
                # 具体解释如下：mode == "boundaries"：mode是一个字符串变量，用于指定计算指标时采用的模式，此处指定的是"boundaries"模式，表示只要求模型预测的实体位置与标注数据中实体位置相交即可。
                # pred_rels：用于存储模型预测出来的某个关系类型rel_type的关系二元组，其中每个元素是一个由(head, tail)组成的元组，表示一个关系实例，head和tail分别表示实体的位置。
                # gt_rels：用于存储标注数据中的某个关系类型rel_type的关系二元组，格式同pred_rels。
                # (rel["head"], rel["tail"])：这行代码提取了一个关系元素rel中的实体位置信息，并将其组成一个关系二元组，存储到pred_rels或gt_rels中。具体来说，rel["head"]表示关系中头实体的位置，rel["tail"]表示关系中尾实体的位置。
                # if rel["type"] == rel_type：这个条件语句用于判断当前遍历到的关系元素 rel 是否属于指定的关系类型 rel_type，如果是，则将其加入到 pred_rels 或 gt_rels 中。
            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)
                    # scores[rel_type]["tp"]：scores是一个字典，其中每个键是一个关系类型，每个值又是一个字典，用于存储该关系类型的各项指标。
                    # 这里scores[rel_type]["tp"]表示关系类型为rel_type的关系在模型预测结果中和标注数据中都被正确预测的个数，tp是true positive的缩写。
                    #
                    # scores[rel_type]["fp"]：表示关系类型为rel_type的关系在模型预测结果中被预测为该关系类型但在标注数据中不属于该关系类型的个数，fp是false positive的缩写。
                    #
                    # scores[rel_type]["fn"]：表示关系类型为 rel_type的关系在标注数据中是该关系类型但在模型预测结果中被预测为其他关系类型或未被预测的个数，fn是false negative的缩写。
                    #
                    # len(pred_rels & gt_rels)：表示关系类型为rel_type 的关系在模型预测结果中和标注数据中都被正确预测的个数，即true positive，计算方式是求出模型预测结果和标注数据的交集，并取其长度。
                    #
                    # len(pred_rels - gt_rels)：表示关系类型为rel_type的关系在模型预测结果中被预测为该关系类型但在标注数据中不属于该关系类型的个数，即false positive，计算方式是求出模型预测结果和标注数据的差集，并取其长度。
                    #
                    # len(gt_rels - pred_rels)：表示关系类型为rel_type的关系在标注数据中是该关系类型但在模型预测结果中被预测为其他关系类型或未被预测的个数，即false  negative，计算方式是求出标注数据和模型预测结果的差集，并取其长度。

    # Compute per entity Precision / Recall / F1
            #这是一个用于计算分类模型评估指标的代码片段，其中包括了计算准确率（precision）、召回率（recall）和F1分数（F1score）的代码。
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0
            # 其中，for 循环迭代 scores 字典中的每个关系类型（rel_type），并根据该关系类型的真阳性（tp）、假阳性（fp）和假阴性（fn）数目计算准确率、召回率和 F1 分数。
            # 如果某个关系类型没有真阳性，那么准确率和召回率都被设置为 0。
        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = (
                2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
            )
        else:
            scores[rel_type]["f1"] = 0
            #在计算F1分数时，如果准确率和召回率之和为0，那么F1分数也被设置为0。否则，根据 F1分数的公式计算F1分数并更新scores字典中对应的值。

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])
        #这些总体评估指标可以用于评估模型在多个类别上的表现，并计算出模型的整体准确性、召回率和F1分数等评估指标。

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

        # 如果真阳性（tp）的数量大于0，则通过tp、fp和fn的数量计算准确率、召回率和F1分数，并将这些值分别赋值给precision、recall和f1变量。
        # 如果tp的数量为0，则说明模型没有正确识别任何正样本，此时准确率、召回率和F1分数都被设置为0。

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn
    # 通过将precision、recall和f1变量的值分别赋值给scores字典中“ALL”关系类型的对应指标来更新总体评估指标。
    # 同时，通过将tp、fp和fn变量的值分别赋值给“ALL”关系类型的对应指标来更新总体评估指标中的总体真阳性、总体假阳性和总体假阴性的数量。
    # 这些总体评估指标可以用于评估模型在所有样本上的表现，并作为比较不同模型表现的指标。

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])
        # 在这个代码片段中，通过使用NumPy库中的mean函数计算所有关系类型F1分数、准确率和召回率的平均值，然后将这些平均值赋值给总体评估指标中的宏平均F1分数、宏平均准确率和宏平均召回率。
        # 宏平均F1分数是一种无偏的指标，能够平衡地衡量每个关系类型的性能，即使在类别分布不平衡的情况下也能提供有用的信息。
        # 这个指标是一种常用的评估指标，可用于比较不同模型之间的表现。
    logger.info(f"RE Evaluation in *** {mode.upper()} *** mode")

    logger.info(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(
            n_sents, n_rels, n_found, tp
        )
    )
        # 这是一个用于在日志中记录关系抽取任务评估结果的代码片段，其中使用了Python中的f - string，它允许在字符串中包含变量和表达式。
        # 在这个代码片段中，使用logger对象将一个INFO级别的日志信息写入日志文件中。
        # 这个日志信息包括了任务评估的模式（mode）、处理的句子数量（n_sents）、实体对数量（n_rels）、正确预测的关系对数量（tp）以及在测试集中发现的关系对数量（n_found）等信息。
        # 在这个日志信息中，通过使用f - string将这些变量和表达式嵌入到字符串中。
        # 通过这个日志信息，我们可以了解到模型在测试集上的表现情况，例如正确预测的关系数量、测试集中实际存在的关系数量、处理的句子数量等等。
        # 这些信息可以用来评估模型在实际应用场景下的表现，并帮助我们优化模型的性能。
    logger.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(scores["ALL"]["tp"], scores["ALL"]["fp"], scores["ALL"]["fn"])
    )
    logger.info("\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(precision, recall, f1))
    logger.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"], scores["ALL"]["Macro_r"], scores["ALL"]["Macro_f1"]
        )
    )
        # 这段代码用于将任务评估的结果写入日志文件中，包括了总体评估结果和macro F1分数。
        # 在这个代码片段中，使用logger对象将三个INFO级别的日志信息写入日志文件中。
        # 第一个日志信息记录了总体的评估结果，包括预测的正确关系数量、预测的错误关系数量以及测试集中未能正确预测的关系数量。
        # 第二个日志信息记录了微平均（micro - average）的评估结果，包括了精确率、召回率和F1分数。
        # 第三个日志信息记录了宏平均（macro - average）的评估结果，包括了宏精确率、宏召回率和宏F1分数。
        # 通过这些日志信息，我们可以了解模型在测试集上的总体表现、每个关系类型的评估结果以及宏平均和微平均的评估结果。
        # 这些信息可以用于评估模型在不同情况下的表现，例如在处理某些特定类型的关系时的表现，或者在不同数据集上的表现等等。
    for rel_type in relation_types:
        logger.info(
            "\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
                rel_type,
                scores[rel_type]["tp"],
                scores[rel_type]["fp"],
                scores[rel_type]["fn"],
                scores[rel_type]["p"],
                scores[rel_type]["r"],
                scores[rel_type]["f1"],
                scores[rel_type]["tp"] + scores[rel_type]["fp"],
            )
        )

    return scores
    # 这段代码是用于打印关系抽取的评估结果，包括所有关系类型的评估结果和总体评估结果。
    # 针对每个关系类型，打印该类型的TP、FP、FN数，以及其precision、recall和f1值，同时也打印其TP+FP的总数。
    # 针对所有关系类型，打印总的TP、FP、FN数，以及其precision、recall和f1值，以及宏平均的precision、recall和f1值。
    # 该函数的返回值是一个字典，包含所有关系类型的评估结果以及总体评估结果。