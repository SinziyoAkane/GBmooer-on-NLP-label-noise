# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)

from datasets import *
from model import Bert4Classify
from trainer import SelfMixTrainer


def print_args(args):
    # print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    # print('--------args----------\n')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    # Huggingface's original arguments
    pretrained_model_name_or_path: Optional[str] = field(
        default='./chinese-bert-wwm-ext',
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    temp: float = field(
        default=0.5,
        metadata={"help": "Temperature for sharpen function"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="CHN",
        metadata={"help": "Name of dataset"}
    )
    train_file_path: Optional[str] = field(
        default="./data/CHN/train_40.0_idn.csv",
        metadata={"help": "The train data file (.csv)"}
    )
    eval_file_path: Optional[str] = field(
        default="./data/CHN/test_clean.csv",
        metadata={"help": "The eval data file (.csv)"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size"}
    )
    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization. Sequences longer."
        },
    )


@dataclass
class OurTrainingArguments:
    seed: Optional[int] = field(
        default=1,
        metadata={"help": "Seed"}
    )
    warmup_epochs: Optional[int] = field(
        default=100,
        metadata={
            "help": "Number of epochs to warmup the model"
                    "only one of the warmup_epochs and warmup_samples should be specified"
        }
    )
    lambda_r: float = field(
        default=0.3,
        metadata={"help": "Weight for R-Drop loss"}
    )
    noise_ratio: int = field(
        default=20,
        metadata={"help": "data noise ratio"}
    )
    noise_type: str = field(
        default="idn",
        metadata={"help": "data noise type"}
    )
    train_epochs: int = field(
        default=20,
        metadata={"help": "Mix-up training epochs"}
    )
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    min_synonyms: int = field(
        default=5,
        metadata={"help": "min elements in one gb"}
    )
    model_save_path: Optional[str] = field(
        default="./checkpoints/cluster",
        metadata={"help": "The path to save model"}
    )
    model_load_path: Optional[str] = field(
        default="./checkpoints/pretrained/CHN_100_100_.pth",
        metadata={"help": "The path to save model"}
    )


def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    log_file = (f'./logs/temp_{data_args.dataset_name}_{training_args.train_epochs}'
                f'_noise_ratio={training_args.noise_ratio}_noise_type={training_args.noise_type}_lr={training_args.lr}'
                f'_min_synonyms={training_args.min_synonyms}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Model Parameters %s", model_args)
    logging.info("Data Parameters %s", data_args)
    logging.info("Training Parameters %s", training_args)

    print_args(data_args)
    print_args(training_args)

    set_seed(training_args.seed)

    train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)

    assert train_num_classes == eval_num_classes
    model_args.num_classes = train_num_classes

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)

    selfmix_train_data = SelfMixData(data_args, train_datasets, tokenizer)
    selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)

    model = Bert4Classify(
        model_args.pretrained_model_name_or_path,
        model_args.dropout_rate,
        model_args.num_classes
    )

    model = model.to(device)

    model.load_model(training_args.model_load_path)

    trainer = SelfMixTrainer(
        model=model,
        train_data=selfmix_train_data,
        eval_data=selfmix_eval_data,
        train_sample=True,
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        rank=0
    )

    test_best, test_last_l = trainer.dynamic_train()

    print('test_best', test_best)
    print('test_last', test_last_l)

    print("Test best %f , last %f" % (test_best, test_last_l[-1]))

    logging.info(f"Best: {str(test_best)} | Acc list: {str(test_last_l)}")

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("current cuda available: ", torch.cuda.device_count())

    run()

if __name__ == '__main__':
    main()
