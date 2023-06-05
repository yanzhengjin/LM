# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer


_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""
             # 定义了一个名为XFUNConfig的类，继承自datasets.BuilderConfig。XFUNConfig类的作用是为XFUN数据集提供配置信息。

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs
            # __init__方法定义了XFUNConfig类的初始化函数。该函数接受一个lang参数，表示输入文本所用的语言，并将其保存到self.lang属性中。additional_langs参数表示可选的其他语言，
            # 如果提供了这个参数，将被保存到self.additional_langs属性中。 ** kwargs参数表示一个包含所有未命名参数的字典。在这个方法中，这些参数将被转发给datasets.BuilderConfig类的初始化函数。


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""
        # 这是一个名为"XFUN"的数据集，继承自datasets.GeneratorBasedBuilder类，用于生成语言数据集。

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]
        # 这是一个名为"XFUN"的数据集，继承自datasets.GeneratorBasedBuilder类，用于生成语言数据集。XFUN数据集有多个配置，每个配置都由一个名字和语言代码构成，这里使用了一个名为xfun.
        # {lang}的格式字符串，其中{lang}会被具体的语言代码替换。_LANG可能是一个语言代码列表。

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        # AutoTokenizer.from_pretrained("xlm-roberta-base")是Hugging Face Transformers库中的一个函数，用于自动加载指定模型的预训练分词器。
        # 在这个例子中，使用了名为xlm - roberta - base的预训练模型。这意味着tokenizer是一个分词器对象，可以用于将原始文本转换为模型输入所需的标记序列。

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )
            # 这是XFUN数据集的一个方法_info，它返回一个datasets.DatasetInfo对象，其中包含了关于数据集的详细信息。
            # DatasetInfo对象包含两个主要的属性：
            # features：描述了数据集中每个样本的特征。在这个例子中，数据集包含了多个特征，包括：
            # "id"：一个字符串，用于标识每个样本的唯一ID。
            # "input_ids"：一个整数序列，表示样本的文本经过分词器处理后的标记序列。
            # "bbox"：一个二维整数序列的序列，表示图像中每个实体的边界框（bounding
            # box）的坐标。
            # "labels"：一个标签序列，表示每个标记的标签，可能是7种类别之一（"O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"）。
            # "image"：一个三维无符号整数数组，表示图像的像素值。
            # "entities"：一个实体序列，表示文本中的实体，包括实体的起始位置、结束位置和标签。
            # "relations"：一个关系序列，表示实体之间的关系，包括关系的头尾实体、开始位置和结束位置。
            # supervised_keys：一个元组或None，指定监督学习任务的输入和输出键名。在这个例子中，由于数据集包含多个特征，因此将supervised_keys设置为None，表示数据集没有明确的监督学习任务。

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
            # 这是XFUN数据集的一个方法_split_generators，它返回一个SplitGenerator列表，其中每个SplitGenerator表示一个数据集划分（例如训练集、验证集、测试集等），并指定用于生成该划分的参数。
            #在这个例子中，_split_generators方法首先定义了要下载的数据集文件的URL。对于训练集和验证集，每个语言的文件名都由语言代码和数据集名称（如"train"）组成

        downloaded_files = dl_manager.download_and_extract(urls_to_download)  #dl_manager对象用于下载并解压缩这些文件，返回一个包含已下载文件路径的字典。
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])
            #如果指定了additional_langs参数，则下载并添加指定语言的训练集文件路径。
        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]
            # 最后，该方法返回一个SplitGenerator列表，包含三个元素，分别表示训练集、验证集和测试集。
            # 每个SplitGenerator对象包含一个name属性，用于标识数据集划分的名称，以及一个gen_kwargs属性，它是一个包含用于生成该划分的参数的字典。
            # 在这个例子中，每个划分都接受一个名为filepaths的参数，它是一个包含训练集文件路径的列表（对于验证集和测试集，文件路径列表只包含一个元素）。

    def _generate_examples(self, filepaths):
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            for doc in data["documents"]:       #代码是在处理一个数据集中的文档，并对每个文档中的图像进行处理，包括加载图像和提取文本中的实体和关系。
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                        #doc["img"]["fpath"]是图像文件的路径，通过os.path.join函数将其与filepath[1]（文件夹路径）组合成完整的图像文件路径。load_image函数加载图像并返回图像数组和大小信息。
                         # document = doc["document"]是文档的文本内容。
                         # tokenized_doc是一个字典，包含三个键值对，分别是"input_ids"、"bbox"和"labels"。这些键值对是用于存储已经经过词汇化和标记化的文本以及它们在图像中的边框坐标和标签。
                         #  entities和relations分别是文档中的实体和关系列表。id2label是一个字典，用于将实体和关系的ID映射到对应的标签。entity_id_to_index_map是一个字典，用于将实体ID映射到在实体列表中的索引位置。
                        #empty_entity是一个空实体集合。
                for line in document:
                    #用于处理文档中的每一行数据。它的主要作用是将文本数据和其对应的视觉数据（bounding box）转换为模型可以接受的格式，并将它们添加到一个已有的数据集中。这些转换包括：
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                            #如果文本数据为空，将该行数据的id添加到 empty_entity集合中，并跳过此行的处理。
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]]) #将该行数据的 id 和对应的标签添加到 id2label 字典中。将该行数据的 linking 信息添加到关系列表 relations 中。
                    tokenized_inputs = self.tokenizer(
                        line["text"],
                            #line["text"]是一个字符串变量，表示需要进行处理的文本内容。
                        add_special_tokens=False,
                            #add_special_tokens是一个布尔变量，表示是否在文本序列的开始和结束位置添加特殊符号，如[CLS]和[SEP]。这里设置为False，即不添加特殊符号。
                        return_offsets_mapping=True,
                            #return_offsets_mapping是一个布尔变量，表示是否返回文本序列中每个字符在原始文本中的位置。这里设置为True，即返回位置信息。
                        return_attention_mask=False,
                            #return_attention_mask是一个布尔变量，表示是否返回序列的attention mask，用于标记哪些位置是需要被attention机制关注的。这里设置为False，即不返回attention mask。
                    )
                      #段代码处理后返回了一个tokenized_inputs变量，该变量是一个字典类型，包含以下键值对：input_ids：表示将输入文本转换为的整数编码序列。
                      # offset_mapping：表示每个token在原始文本中的起始位置和结束位置，即字符级别的位置信息。
                      # text_length：表示输入文本的长度（字符数）。
                      # ocr_length：表示OCR文本的长度（字符数）。
                      # bbox：表示OCR文本对应的边界框位置信息。
                      # last_box：表示OCR文本对应的最后一个边界框的位置信息。
                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    last_box = None
                        # input_ids：表示将输入文本转换为的整数编码序列。
                        # offset_mapping：表示每个token在原始文本中的起始位置和结束位置，即字符级别的位置信息。
                        # text_length：表示输入文本的长度（字符数）。
                        # ocr_length：表示OCR文本的长度（字符数）。
                        # bbox：表示OCR文本对应的边界框位置信息。
                        # last_box：表示OCR文本对应的最后一个边界框的位置信息。
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        # 这段代码使用了Python中的zip()函数来同时迭代两个列表tokenized_inputs["input_ids"]和tokenized_inputs["offset_mapping"]，并将它们中对应位置的元素打包成一个元组。
                        # tokenized_inputs是一个被分词器(tokenizer)处理过的文本输入。其中，input_ids是输入文本被转换成对应的token ID序列，offset_mapping是输入文本中每个字符对应的token的起始和结束位置。
                        # 例如，如果输入文本是"Hello, world!"，那么offset_mapping可能会被处理成[(0, 5), (6, 12), (12, 13)]，其中(0, 5)
                        # 表示对应着"Hello"这个token的起始位置和结束位置分别是0和5。
                        # 在这段代码中，token_id和offset分别代表了当前迭代到的input_ids和offset_mapping中的元素，可以用来进一步处理和分析分词结果。
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                            # 这段代码是在处理OCR(Optical Character Recognition)识别结果，将识别出的文本和其对应的边框信息按照token进行对齐。
                            # 如果当前迭代到的input_ids的值为6，说明对应的token是一个标记为PAD(填充)的无意义token，这里会将其对应的bbox(边框)
                            # 设置为None并继续下一个迭代。如果当前迭代到的token不是PAD，则将其对应的字符长度offset[1] - offset[0]加到text_length中，用于后续与OCR识别结果的字符长度进行比较。
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                            # 接着，使用一个while循环不断从OCR识别结果中pop出单词，将其对应的字符长度加到ocr_length中，并将单词的边框信息simplify_bbox(ocr_word["box"])
                            # 添加到tmp_box列表中。其中simplify_bbox可能是一个对OCR识别结果中边框信息进行简化处理的函数，具体实现可能因应用场景不同而不同。
                            # 当ocr_length达到或超过text_length时，说明已经找到了与当前token对应的OCR文本和边框信息，此时将tmp_box添加到bbox列表中并跳出while循环，继续下一个迭代。
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box
                            # 这段代码是用来处理OCR识别结果和token对齐后的边框信息。其中，tmp_box是前面代码段中存储当前token对应的OCR识别结果边框信息的列表。
                            # 如果在处理当前token时，OCR识别结果为空，即len(tmp_box) == 0，则将tmp_box设置为上一个token对应的OCR识别结果边框信息last_box。
                            # 这样可以避免出现空的边框信息，保证最后生成的bbox列表中每个元素都包含了一个有效的边框信息。
                            # 接下来，将tmp_box中的所有边框信息合并成一个整体的边框，并将其归一化(normalize_bbox)为相对于整张图片的比例坐标，最后将归一化后的边框信息添加到bbox列表中。
                            # last_box则用来存储上一个token对应的OCR识别结果的边框信息，以便在下一个token对齐时，如果当前token的OCR识别结果为空，则使用last_box中存储的上一个token的OCR识别结果的边框信息。
                            # 这样可以避免在处理过程中出现空的边框信息，确保最终的边框列表中每个元素都有一个有效的边框信息。
                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]
                        # 这段代码是对边框列表bbox中的元素进行处理。如果列表中的某个元素为None，则将其设置为该元素后面一个元素的坐标(x1, y1, x2, y2)，
                        # 即让None元素的坐标与后面一个非None元素的坐标相同，这样可以确保最终的边框列表中没有None元素。
                        # 代码中使用了一个列表推导式和enumerate函数来遍历列表中的元素和下标，对于每个元素，如果其值为None，则将其替换为其后面一个元素的坐标，否则不做任何处理，最后得到一个新的边框列表bbox。
                    if line["label"] == "other":
                        label = ["O"] * len(bbox)
                    else:
                        label = [f"I-{line['label'].upper()}"] * len(bbox)
                        label[0] = f"B-{line['label'].upper()}"
                        # 这段代码是根据OCR识别结果所对应的文本标签，生成一个标签列表label。具体实现方式如下：
                        # 首先判断当前OCR识别结果对应的文本标签是否为"other"，如果是，则将label列表中所有元素的值都设为 "O"，表示这些元素对应的文本都不属于任何标签。
                        # 如果当前OCR识别结果对应的文本标签不为"other"，则将label列表中所有元素的值都设为"I-文本标签"，其中"文本标签"
                        # 为当前OCR识别结果所对应的文本标签，且所有文本标签均转换为大写字母。接着将label列表中的第一个元素的值设为"B-文本标签"，表示这个元素对应的文本是一个标签的开头。
                        # 最后得到的label列表中每个元素表示对应边框所对应的文本的标签。
                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                        # 这段代码是将OCR识别结果对应的边框信息和文本标签信息添加到tokenized_inputs字典中。
                        # 其中，tokenized_inputs是一个包含tokenized文本信息的字典，通过调用update方法，将字典中的信息进行更新。bbox是一个列表，存储了OCR识别结果所对应的边框信息。
                        # label是一个列表，存储了OCR识别结果所对应的文本标签信息。通过将这两个列表添加到tokenized_inputs字典中，并分别用"bbox"和"labels"作为键，实现了将这些信息与tokenized文本信息进行绑定。
                        # 这样，在后续的模型训练和评估过程中，就可以使用这些信息来训练模型，提高模型的性能。
                    if label[0] != "O":
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),
                            }
                        )
                        # 这段代码是在生成NER（Named Entity Recognition）训练数据时，处理每个OCR识别结果对应的实体信息。
                        # 首先判断当前OCR识别结果对应的标签列表label的第一个元素是否为"O"，如果不是，则说明这个OCR识别结果对应的文本是一个实体的开头。
                        # 接着在entity_id_to_index_map字典中添加一个映射，将OCR识别结果的id映射到对应实体在entities列表中的索引位置。
                        # 最后，将一个新的实体字典添加到entities列表中，该字典包含了实体在tokenized文本中的起始位置、终止位置和标签信息。
                        # 具体来说，新的实体字典包含三个键值对，分别是"start"、"end"和"label"。
                        # 其中，"start"键的值为当前tokenized文本中的输入ID列表的长度，即当前OCR识别结果对应的文本在tokenized文本中的起始位置。
                        # "end"键的值为当前tokenized文本中的输入ID列表的长度加上当前OCR识别结果的tokenized输入ID列表的长度，即当前OCR识别结果对应的文本在tokenized文本中的终止位置。
                        # "label"键的值为当前OCR识别结果对应的文本的标签，转换为大写字母后的结果。
                        # 这样，在后续的NER模型训练和评估过程中，可以使用这些实体信息来训练模型，并对模型进行评估。
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
                            # 这段代码是将OCR识别结果的tokenized输入信息添加到完整文本的tokenized输入信息中。
                            # 首先遍历完整文本的tokenized输入信息tokenized_doc中的每一个元素，并对其进行更新。
                            # 对于每一个元素，都将其原本的值与OCR识别结果的tokenized输入信息tokenized_inputs对应元素的值拼接起来，并将其更新到tokenized_doc字典中。
                            # 这样，就将OCR识别结果的tokenized输入信息添加到完整文本的tokenized输入信息中，生成了最终的tokenized输入信息。
                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                    # 这段代码用于处理OCR识别结果中的实体关系信息。其中，OCR识别结果中的实体关系信息包括了实体之间的关系类型和关系所连接的实体。
                    # 该信息以一个由二元组构成的列表形式存储，其中每个二元组表示一个实体关系，其第一个元素为连接实体1的id，第二个元素为连接实体2的id。
                    # 首先，将OCR识别结果中的所有实体关系二元组存储到relations列表中，并去重。
                    # 接着，使用列表推导式筛选出那些连接实体id不为空的实体关系，并将其存储到relations列表中。
                    # 最后，初始化kvrelations列表，用于存储实体关系转化后的信息，该列表的每个元素表示一个实体关系，其中包含了实体1、实体2、实体关系类型以及实体关系id等信息。
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue
                        # 这段代码用于将OCR识别结果中的实体关系信息转换成KV（key - value）对关系信息，以便后续模型训练使用。
                        # 在该代码中，遍历了OCR识别结果中的所有实体关系，并对每个实体关系进行如下处理：
                        # 首先，通过实体id到实体类型的映射id2label获取连接实体的类型信息。如果连接的实体类型是"question"和"answer"，则将该实体关系转化成KV对关系信息，并存储到kvrelations列表中。
                        # 具体而言，将实体1和实体2的id转化成实体1和实体2在实体列表entities中的下标，并将其存储到KV对关系的"head"和"tail"字段中，用于表示关系的起点和终点。
                        # 如果连接的实体类型不是"question"和"answer"，则忽略该实体关系。
                        # 最终，kvrelations列表中存储了OCR识别结果中的所有实体关系的KV对关系信息。

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)
                    # 该函数用于获取视化分析。
                    # 具体而言，该函数接受一个KV对关系作为输入rKV对关系中两个实体组成的关系所对应的文本在输入中的起始和结束位置，以便后续用于可el，并分别获取KV对关系中两个实体的开始和结束位置，将其存储到列表bound中。
                    # 最终，函数返回bound列表中的最小值和最大值，用于表示KV对关系对应的文本在输入中的起始和结束位置。

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )
                    # 该代码块的作用是对KV对关系进行排序。具体而言，该代码块首先遍历KV对关系列表kvrelations中的每一个关系rel，并通过调用函数get_relation_span获取该关系对应的文本在输入中的起始和结束位置，
                    # 并将结果存储到一个字典中。字典中的键"head"和"tail"分别对应于KV对关系中的头实体和尾实体，而键"start_index"和"end_index"则分别对应于该关系在输入中的起始和结束位置。
                    # 随后，该代码块使用Python内置的sorted函数，以头实体的索引为排序关键字，对字典列表进行排序。
                    # 最终，该代码块返回的是一个已经按照头实体索引排序好的字典列表relations。
                chunk_size = 512
                    # 这行代码定义了一个整数变量chunk_size，用于指定每个输入文本片段的最大长度。
                    # 在自然语言处理中，通常需要将较长的文本划分成多个较短的片段进行处理，以避免模型训练过程中内存溢出等问题。
                    # 因此，该代码中的chunk_size表示每个文本片段的最大长度，超过该长度的文本将被切割成多个片段，每个片段长度不超过chunk_size。
                    # 具体而言，可以将输入文本切分成多个长度为chunk_size的子串，将这些子串分别送入模型中进行处理，最终将所有处理结果进行合并。
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                        # 这段代码定义了一个for循环，其中变量chunk_id表示当前处理的文本片段的编号，变量index表示当前文本片段的起始位置在原始文本中的索引。
                        # 循环的范围是从0到len(tokenized_doc["input_ids"])，每次递增chunk_size。这里的chunk_size是在前面定义的，用于指定每个文本片段的最大长度。
                        # 循环中的range(0,len(tokenized_doc["input_ids"]),chunk_size)实际上是将输入文本切分成多个长度为chunk_size的子串，每次循环处理一个子串。
                        # 具体而言，循环中的enumerate函数可以将一个可迭代对象的每个元素和对应的索引配对，并返回一个迭代器。
                        # 在这里，enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size))的作用是将每个文本片段的编号和对应的起始位置配对，方便后续对每个片段进行处理。
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index : index + chunk_size]
                    entities_in_this_span = []
                    global_to_local_map = {}
                            # 这段代码应该是对原始文本进行分块处理，每个块的大小为512。
                            # 循环遍历了整个文本，每次处理一个大小为512的块，将处理后的结果保存在item字典中。
                            # 然后根据当前块中存在的实体的位置，更新entities_in_this_span列表和global_to_local_map字典。
                            # entities_in_this_span列表记录了当前块中存在的实体，global_to_local_map字典用来记录当前块中的实体在全局实体列表中的索引。
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)
                    # 这段代码的作用是对每个chunk进行处理，提取出当前chunk中的文本和实体，并将实体的位置调整为相对于当前chunk的位置。
                    # 具体来说，首先创建一个空字典item用于存储当前chunk的信息，包括tokenized_doc中每个键对应的值。
                    # 然后遍历所有的实体，找到在当前chunk中的实体，并将实体的位置调整为相对于当前chunk的位置，将实体存储在entities_in_this_span列表中。
                    # 此外，还创建了一个字典global_to_local_map，用于记录在当前chunk中的实体在entities_in_this_span中的索引，以便在构建关系时进行使用。
                    relations_in_this_span = [] #此行初始化一个名为relations_in_This_span的空列表。这表明，下面的代码将用当前正在处理的输入文本块中存在的关系填充此列表。
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )
                            # 这段代码是对于每个文档的分块进行处理。首先创建一个空的字典item，然后将每个键值对k: tokenized_doc[k][index: index + chunk_size]加入该字典中。
                            # 接着，将所有在该分块中出现的实体提取出来，将这些实体加入到entities_in_this_span中，并通过global_to_local_map将实体的全局id映射到该分块中的局部id。
                            # 最后，将所有在该分块中出现的关系提取出来，并将这些关系加入到relations_in_this_span中。
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item
                    # 这段代码是一个生成器函数，用于生成处理后的文本数据。具体来说，它按照chunk_size（512）将长文本切分成若干段，每段都是一个包含tokenized_doc中各种属性的字典item。
                    # 每个item字典还包括了id、image、entities和relations四个属性，分别表示处理后的文本段的ID、对应的图片、实体列表和关系列表。
                    # 函数通过yield 语句一个一个地返回这些字典，从而形成一个生成器对象。这个生成器对象可以用于迭代获取处理后的文本数据。
