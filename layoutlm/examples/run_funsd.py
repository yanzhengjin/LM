#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import layoutlmft.data.datasets.funsd
import transformers
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import FunsdTrainer as Trainer
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    #HfArgumentParser函数接受三个参数：ModelArguments，DataTrainingArguments和TrainingArguments，它们分别代表模型参数，数据训练参数和训练参数。
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        #如果我们只向脚本传递一个参数，它是一个json文件的路径，让我们解析它以获取参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        #这句代码的意思是，使用parser.parse_json_file函数从指定的json文件中解析出model_args、data_args和training_args参数，并将它们赋值给相应的变量。
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        #parser.parse_args_into_dataclasses()意思是将命令行参数解析为数据类，其中model_args、data_args和training_args是三个数据类，用于存储命令行参数的值。

    # Detecting last checkpoint.
    #正在检测上一个检查点。
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #如果输出目录存在，并且training_args.do_train为真，并且training_args.overwrite_output_dir为假，则意味着不能覆盖输出目录。
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        #get_last_checkpoint(training_args.output_dir)意思是从训练参数的输出目录中获取最后一个检查点。
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        #如果最后一个检查点为空，并且输出目录中的文件数量大于0，则意味着已经有训练过程的输出文件存在。
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    #设置日志记录是指在计算机系统中记录系统活动的过程。它可以帮助您更好地了解系统的行为，并帮助您诊断和解决问题。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 这是一段Python代码，用于设置日志记录的基本配置和级别。logging.basicConfig()
    # 是Python内置的函数，用于设置全局日志记录的基本配置，比如日志记录的格式、日期格式、处理程序等。其中，format参数指定日志记录的格式，datefmt参数指定日期格式，handlers参数指定处理程序，这里使用logging.StreamHandler(sys.stdout)
    # 将日志输出到标准输出流。logger.setLevel()
    # 方法用于设置日志记录器的级别，即设置日志记录器记录的日志级别。如果is_main_process(training_args.local_rank)
    # 返回True，则将日志级别设置为logging.INFO，否则设置为logging.WARN。这里的逻辑是，如果是主进程，则记录INFO级别的日志，否则记录WARN级别的日志。
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    #这句代码的意思是，如果training_args.local_rank是主进程，则将logger的日志级别设置为logging.INFO，否则将其设置为logging.WARN。

    # Log on each process the small summary:
    #Logoneachprocessthesmallsummary是指在每个进程中记录一个小的摘要。
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    #设置Transformers日志记录器（仅限主进程）的详细程度为info。
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    #这些代码的意思是，如果training_args.local_rank是主进程，则设置日志输出的详细程度为“信息”，并启用默认处理程序和显式格式。
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    #设置种子是指在初始化模型之前，使用随机数生成器来设置一个固定的随机数种子，以便每次运行模型时都能够得到相同的结果。
    set_seed(training_args.seed)
    #set_seed（）是一个为Human对象设置随机种子的函数。这用于确保每次运行程序时生成相同的随机数。training_args.seed参数是用于设置随机种子的种子值。

    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.funsd.__file__))
    #这句代码的意思是加载funsd数据集，并将其路径作为参数传递给load_dataset函数。
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
        # 根据training_args.do_train的值来决定读取哪个数据集的列名和特征。
        # 如果training_args.do_train为True，则使用训练集（datasets["train"]）的列名和特征；如果training_args.do_train为False，则使用验证集（datasets["validation"]）的列名和特征。
        # 这些列名和特征是用于数据集的读取和处理。具体来说，column_names包含数据集的列名，features包含数据集中每个示例的特征描述信息。例如，对于文本分类任务，features可能包含每个示例的文本输入和标签输出。
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )
    # 如果"tokens"在column_names中，则文本列的名称为"tokens"，否则文本列的名称为column_names列表的第一个元素。
    # 如果f"{data_args.task_name}_tags"在column_names中，则标签列的名称为f"{data_args.task_name}_tags"，否则标签列的名称为column_names列表的第二个元素。
    # 例如，如果column_names列表为["sentences", "ner_tags"]，并且data_args.task_name为 "ner"，
    # 则文本列的名称为"sentences"，标签列的名称为"ner_tags"。如果column_names列表为["text"]，则文本列的名称为"text"，标签列的名称为"ner_tags"（假设data_args.task_name为"ner"）。
    remove_columns = column_names
    #remove_columns=column_names意思是从数据集中删除指定的列，其中column_names是要删除的列的名称列表。

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the unique labels.
    #如果标签不是“Sequence[ClassLabel]”，我们需要遍历数据集以获取唯一标签。
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        #unique_labels=unique_labels|set(label)意思是将标签列表中的标签添加到唯一标签集合中，以便可以查看唯一标签集合中的所有标签。
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    #这个函数的作用是从一组标签中获取唯一的标签列表，并将其排序。
    
    if isinstance(features[label_column_name].feature, ClassLabel):
    #如果features[label_column_name].feature是ClassLabel类的实例，则意味着它是一个类标签，可以用来标记数据集中的每一行。
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        #无需转换标签，因为它们已经是int。
        label_to_id = {i: i for i in range(len(label_list))}
        #这句话的意思是，如果i在label_list的长度范围内，就将label_list中的第i个标签映射到id上。
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        #这句话的意思是，将标签列表中的每个标签映射到一个唯一的ID，其中l是标签，i是ID。
    num_labels = len(label_list)
    #意思是获取标签列表中标签的数量，即标签的总数。

    # Load pretrained model and tokenizer(加载预训练模型和"Tokenizer" 表示将文本分割成单词或子单词的工具。例如，将一段英文句子分割成单词的tokenizer会将句子"Hello, how are you?"分割成"Hello", ",", "how", "are", "you", "?"这些单词。)
    
    # Distributed training:（分布式训练）
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.（from_pretrained方法保证只有一个本地进程可以同时下载model和vocab。）
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    #这句话的意思是，从预先训练的模型中获取自动配置，其中num_labels参数为标签的数量，finetuning_task参数为数据任务的名称，cache_dir参数为模型缓存目录，revision参数为模型修订版本，use_auth_token参数为是否使用认证令牌。
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
        # 使用了Hugging Face的Transformers库中的AutoTokenizer类，它可以自动选择和加载适当的分词器（tokenizer）模型。
        # from_pretrained()方法指定了要使用的模型名称（model_args.model_name_or_path）或分词器名称（model_args.tokenizer_name），并在需要时从Hugging
        # Face模型中心下载相应的模型 / 分词器。可以通过设置cache_dir参数来指定模型 / 分词器的缓存目录。
        # use_fast参数为True时，将使用Hugging
        # Face的tokenizers库中的快速分词器。
        # revision参数可选，可以指定要使用的模型 / 分词器的版本号。
        # use_auth_token参数可选，如果设置为True，将使用用户的Hugging Face API密钥进行身份验证。
        # 总之，这段代码的作用是加载一个适当的分词器模型并准备进行后续的自然语言处理任务。
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
        # 代码使用Hugging Face Transformers库中的AutoModelForTokenClassification类来创建一个预训练模型。具体来说，它根据给定的模型名称或路径，使用预训练权重加载该模型，并为标记分类任务进行微调。
        # 该函数接受一些参数，例如使用的配置文件、缓存目录、模型版本等。其中，from_tf = bool(".ckpt" in model_args.model_name_or_path)
        # 用于判断模型是否是从TensorFlow检查点加载的，并据此选择适当的权重加载方式。
        # 最后，use_auth_token参数用于验证Hugging Face提供的API token，以便在私有模型仓库中访问私有模型。如果该参数被设置为True，则需要提供有效的API token。
    # Tokenizer check: this script requires a fast tokenizer.一个快速的分词器

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
    #这句话的意思是，如果tokenizer不是PreTrainedTokenizerFast的实例
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy（#预处理数据集 填充策略）
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
        # 是将所有文本分词，并将标签与分词对齐。具体来说，就是将所有文本分割成较小的单元，称为
        # "token"（通常是单词或子词），然后为每个token分配一个标签，表示该token的一些信息，例如其词性或命名实体类型。
        # 例如，如果文本是 "约翰喜欢踢足球"，分词后的token可能是
        # "约翰"、"喜欢"、"踢"、"足球"，标签可能是
        # "人名"、"动词"、"动词"、"运动"。将标签与分词对齐意味着将标签分配给正确的token，以便标签
        # "人名"被分配给"约翰"，"动词"被分配给"喜欢"和"踢"，"运动"被分配给"足球"。这个过程通常用于准备文本数据
    def tokenize_and_align_labels(examples):
                # 这是一个用于处理文本分类数据集的Python函数，它将输入的文本数据集进行分词，并将每个单词的标签对齐。该函数的输入参数examples应该是一个包含文本列和标签列的Pandas DataFrame。
                # 函数的输出是一个字典，其中包含了对输入文本进行分词处理后的结果，以及每个单词对应的标签和边界框信息。
                # 具体来说，函数使用了一个名为tokenizer的分词器对文本进行分词处理，并使用一个名为label_to_id的字典将标签映射到数字ID。
                # 最终，函数返回一个包含标签和边界框信息的字典，该字典可以被传递给机器学习模型进行训练和测试。
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        bboxes = []
        images = []
            # 使用了一个tokenizer对文本数据进行处理，生成了一个包含了tokenized_inputs的字典，其中包括了经过编码的输入文本、对应的attention mask等信息。
            # 具体来说，它的作用是将examples[text_column_name]中的文本数据进行编码，并根据设置的参数进行padding、截断、切分等操作，以生成规整的输入序列。同时，它还使用了is_split_into_words参数来说明输入文本已经被切分为单词列表，并返回了 return_overflowing_tokens
            # 参数，以便对长文本进行处理。在这段代码之后，labels、bboxes和 images三个变量将会用于存储对应的标签、边界框和图像数据，以便后续处理。
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            #tokenized_inputs.word_ids(batch_index=batch_index)意思是从给定的batch_index中提取tokenized_inputs中的word_ids。它可以用来从一个batch中提取特定的word_ids，以便进行进一步的处理。
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]
            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
            # 这段代码是用来将一个batch的数据处理成模型可以接受的格式，包括文本输入数据（input_ids和attention_mask），
            # 以及标签数据（label_ids）和边界框数据（bbox_inputs）。
            # 具体来说，这段代码使用了tokenized_inputs对象中的word_ids方法，从中提取出每个单词对应的索引，然后根据单词索引将标签和边界框数据对应到每个单词上，最终将处理好的数据存储在labels、bboxes和images列表中。
            # 这段代码的主要逻辑是遍历每个单词的索引，然后根据单词的索引和标签列中的值，
            # 将对应的标签和边界框数据添加到label_ids和bbox_inputs列表中。
            # 特别地，对于一些特殊的标记（如padding和CLS、SEP标记），它们的word_id可能为None，此时将对应的label_ids设置为 - 100，以便在计算损失函数时自动忽略它们。
            # 最终，代码将处理好的标签、边界框和图像数据分别存储在labels、bboxes和images列表中，并返回这些数据作为模型的输入。
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        #这句话的意思是，如果data_args.max_train_samples不为空，那么就从训练数据集中选择data_args.max_train_samples个样本进行训练。
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
            # 代码是用来加载和预处理训练数据集的。其中，如果训练参数中指定了要进行训练（--do_train），则会先检查是否提供了训练数据集（train），
            # 如果没有则会抛出一个值错误（ValueError）。然后，将训练数据集赋值给变量train_dataset。
            # 如果指定了max_train_samples参数，则会从整个训练数据集中选择前max_train_samples个样本进行训练。
            # 接下来，使用map方法对训练数据集进行预处理。tokenize_and_align_labels函数是一个自定义的函数，
            # 用来对文本进行分词和标签对齐操作。batched=True参数表示对数据集进行分批处理。
            # remove_columns参数表示要从数据集中删除的列。num_proc参数表示预处理过程中使用的进程数。
            # load_from_cache_file参数表示是否从缓存文件中加载数据。如果overwrite_cache参数为True，则会重新生成缓存文件，否则会从已有的缓存文件中加载数据。

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        #这句话的意思是，如果max_val_samples参数不为None，那么将eval_dataset变量设置为从0到data_args.max_val_samples范围内的数据集。
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        #eval_dataset意思是评估数据集，map函数意味着对数据集中的每个元素执行指定的操作，tokenize_and_align_labels意味着对每个元素进行标记化和标签对齐，
        #batched=True意味着将数据集分成多个批次，remove_columns意味着从数据集中删除指定的列，num_proc意味着使用多少个处理器来执行预处理，load_from_cache_file意味着是否从缓存文件中加载数据。

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        #这句话的意思是，如果data_args.max_test_samples不为None，那么test_dataset将被选择为从0到data_args.max_test_samples的范围内的数据集。
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        #test_dataset=test_dataset.map(tokenize_and_align_labels,batched=True,remove_columns=remove_columns,num_proc=data_args.preprocessing_num_workers,load_from_cache_file=notdata_args.overwrite_cache,)意思是使用tokenize_and_align_labels函数对test_dataset进行映射，batched=True表示使用批处理，remove_columns=remove_columns表示删除指定的列，
        #num_proc=data_args.preprocessing_num_workers表示使用指定数量的处理器，load_from_cache_file=notdata_args.overwrite_cache表示从缓存文件中加载数据，而不是覆盖缓存文件。

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )
    #DataCollatorForKeyValueExtraction是一个用于提取键值对的数据收集器，
    #它使用tokenizer来将输入文本分割成单词，并将其填充到指定的长度（如果training_args.fp16设置为True，则pad_to_multiple_of参数设置为8，否则设置为None），最大长度为512。

    # Metrics
    metric = load_metric("seqeval")
    # load_metric()函数来加载名为seqeval的指标（metric）。具体来说，seqeval是用于序列标注任务的指标库，可以用来评估模型在这种任务上的性能。
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
            # 这个函数的作用是计算模型预测结果的评估指标，其中模型的预测结果和标签被打包为元组p，并且使用了一个外部的指标metric。
            # 该函数使用了NumPy库进行数组的处理和计算，并使用了列表推导式和嵌套列表解析等Python语言特性。
            # 在函数内部，首先通过np.argmax函数找出每个样本的预测结果中概率最大的类别作为最终的预测类别，并将预测结果和标签转换成字符串形式，以便后续计算指标。
            # 接下来，函数会使用label_list中的值将预测结果和标签从数字形式转换成字符串形式，并且会移除特殊的标记（-100），这些标记通常用于指示模型在预测时应该忽略这些标记所代表的部分。
            # 然后，函数使用metric.compute计算预测结果的指标值，并根据data_args.return_entity_level_metrics的取值返回不同的结果。如果该值为True，则会返回每个实体级别的指标值，否则只会返回整体的指标值。

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    #Trainer是一个类，它接受模型、训练参数、训练数据集、评估数据集、令牌化器、数据收集器和计算指标作为参数，用于训练和评估模型。

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        #max_train_samples的意思是：如果data_args.max_train_samples不为空，则max_train_samples等于data_args.max_train_samples，否则等于训练数据集的长度。
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    # 使用了一个名为trainer的对象来对一个模型进行预测。它的作用是：检查training_args.do_predict是否为真，如果是则打印日志信息；
    # 调用trainer.predict(test_dataset)方法进行预测，并将预测结果、标签和指标保存在predictions、labels ,metrics变量中；
    # 对预测结果进行后处理，将预测结果转换为标签列表，并去除标签列表中的特殊标记（在数据预处理时添加）；
    # 使用trainer.log_metrics方法记录模型在测试集上的性能指标，并使用trainer.save_metrics方法保存性能指标；
    # 将处理后的预测结果保存到文件test_predictions.txt中。
    # 在这个代码段中，test_dataset是用于模型预测的测试数据集，label_list是标签列表，output_test_predictions_file是保存预测结果的文件路径。此外，这段代码还使用了
    # Python的os模块来处理文件路径。


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
