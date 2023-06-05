# coding=utf-8

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)

            # 该类定义了一个构造函数
            # __init__，该构造函数接受任意数量的关键字参数（使用 ** kwargs），并将这些参数转发给
            # datasets.BuilderConfig
            # 类的构造函数。在这个特定的例子中，super(FunsdConfig, self).__init__(**kwargs)
            # 调用父类的构造函数，以便在实例化
            # FunsdConfig
            # 类时，父类的构造函数也会被执行。


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]
        # 在该类中，定义了一个名为"BUILDER_CONFIGS"的类变量，它包含一个"FunsdConfig"类的实例。这个"FunsdConfig"实例描述了"Funsd"数据集的名称、版本和描述。

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

        # 代码定义了一个名为_info的函数，该函数返回一个datasets.DatasetInfo对象，其中包括了有关数据集的信息和元数据，如数据集的描述、特征、主页和引用等。具体来说，该数据集包括以下特征：
        #  "id"：一个字符串值，表示文档的唯一标识符。
        #  "tokens"：一个字符串序列，表示文档中每个标记的文本。
        #  "bboxes"：一个整数序列的序列，表示每个标记的边界框（boundingbox）的坐标。每个边界框由四个整数(x_min, y_min, x_max, y_max)表示。
        # "ner_tags"：一个字符串序列的序列，表示每个标记的命名实体识别（Named Entity Recognition，NER）标签。每个标签是七个可能的值之一："O"、"B-HEADER"、"I-HEADER"、"B-QUESTION"、"I-QUESTION"、"B-ANSWER"
        # 和"I-ANSWER"。"image"：一个形状为(3, 224, 224)的三维数组，表示与文档相关的图像。此外，该数据集是一个无监督学习的数据集，因此supervised_keys参数为None。
        # 此外，数据集的主页是https: // guillaumejaume.github.io / FUNSD /，引用信息存储在_CITATION中。


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]
        #代码是Hugging Face的Datasets库中的一个方法，用于定义数据集的切分方式。具体来说，该方法返回一个SplitGenerators列表，其中包含训练集和测试集的切分方式。
        #方法通过dl_manager下载并解压缩了一个名为“dataset.zip”的文件，然后返回了两个SplitGenerator对象，分别代表训练集和测试集。
        # 这两个对象都包含了一个gen_kwargs参数，它们用于传递参数给后续生成切分数据的方法。其中filepath参数表示训练集或测试集的文件路径。

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []        # "tokens": 标注中所有单词的列表，字符串类型。
            bboxes = []        # "bboxes": 标注中所有单词的边界框的列表，由元组组成，每个元组包含四个浮点数，表示左上角和右下角的坐标。
            ner_tags = []      # "ner_tags": 所有标注中所有单词的命名实体标签的列表，字符串类型。

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
                # 在代码中，首先通过调用os模块的join函数来获取JSON文件的路径，然后打开该文件并使用json模块将其中的文本数据加载到data变量中。
                # 接下来，通过将JSON文件的扩展名替换为“png”来获取对应的图像文件路径，并使用load_image函数加载图像和图像尺寸信息。
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))
                    # 接下来的循环用于遍历JSON文件中的每个“item”，其中包含一个单词列表（words）和一个标签（label）。
                    # 如果该item的label是“other”，则将其中的每个单词添加到tokens列表中，并将其对应的NER标签设置为“O”（代表“其他”）。
                    # 如果该item的label不是“other”，则将单词列表中的第一个单词添加到tokens列表中，并将其对应的NER标签设置为“B -” + label（代表“开头”）。
                    # 然后将剩余的单词添加到tokens列表中，并将它们的NER标签设置为“I -” + label（代表“内部”）。此外，还将单词的边界框归一化并添加到bboxes列表中。

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}

                # 用于在迭代过程中逐个生成一些包含OCR处理后的文本和图像信息的数据条目。其中，yield关键字用于生成每个数据条目，并将其作为字典类型返回。这个字典包含以下字段：
                # guid：生成数据条目的唯一ID。
                # id：数据条目的ID，与guid相同。
                # tokens：由OCR处理后的文本数据组成的列表，每个元素都是一个单词。
                # bboxes：一个与tokens列表中的每个单词相对应的边界框坐标列表，这些边界框坐标描述了单词在图像中的位置。
                # ner_tags：一个与tokens列表中的每个单词相对应的命名实体标签列表，表示单词所属的实体类别。
                # image：OCR处理后的图像数据。
