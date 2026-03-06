# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import torch

# from ennhance_sample_pairs import Positive_and_negative_sample_pairs
from torch.utils.data import Dataset, DataLoader
import torch.distributed
import torch.multiprocessing as mp

NUM_CLASSES = {
    'TREC': 6,
    'IMDB': 2,
    'AGNEWS': 4,
    'CHN': 2,
}


class DataToDataset(Dataset):
    def __init__(self, data, contrast=False):
        # true_labels used for drawing figures
        self.labels, self.texts, self.true_labels = data.values[:, 0], data.values[:, 1], data.values[:, 2]
        self.contrast = contrast
        # if self.contrast:
        #     self.positive_texts = np.array(Positive_and_negative_sample_pairs(self.texts))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        if self.contrast:
            return self.texts[index], self.positive_texts[index], self.labels[index], self.true_labels[index]
        else:
            return self.texts[index], self.labels[index], self.true_labels[index]


def load_dataset(data_path, dataset_name, contrast=False):
    extension = data_path.split(".")[-1]
    assert extension == 'csv'

    data = pd.read_csv(data_path, header=None)

    if dataset_name in NUM_CLASSES:
        num_classes = NUM_CLASSES[dataset_name]
    else:
        num_classes = max(data.values[:, 0]) + 1

    logging.info('num_classes is %d', num_classes)
    return DataToDataset(data, contrast), num_classes


class SelfMixDataset(Dataset):
    def __init__(self, data_args, dataset, tokenizer, mode):

        self.data_args = data_args
        self.labels = dataset.labels
        self.inputs = dataset.texts
        self.true_labels = dataset.true_labels

        self.mode = mode
        self.tokenizer = tokenizer

        if self.mode == 'contrast':
            self.positive_inputs = dataset.positive_texts

    def __len__(self):
        return len(self.inputs)

    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.data_args.max_sentence_len, return_tensors='pt')

        for item in tokens:
            tokens[item] = tokens[item].squeeze()

        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def __getitem__(self, index):
        text = self.inputs[index]
        input_id, att_mask = self.get_tokenized(text)
        if self.mode == 'all':
            return input_id, att_mask, self.labels[index], self.true_labels[index], index
        elif self.mode == 'contrast':
            positive_text = self.positive_inputs[index]
            positive_input_id, positive_att_mask = self.get_tokenized(positive_text)

            return (input_id, att_mask, positive_input_id, positive_att_mask,
                    self.labels[index], self.true_labels[index], index)


class SelfMixData:
    def __init__(self, data_args, datasets, tokenizer):
        self.data_args = data_args
        self.datasets = datasets
        self.tokenizer = tokenizer

    def run(self, mode, sampler=True):
        if mode == "all":
            all_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="all")
            if sampler:

                train_sampler = torch.utils.data.DistributedSampler(
                    all_dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True
                )
            else:
                train_sampler = None

            all_loader = DataLoader(
                dataset=all_dataset,
                batch_size=self.data_args.batch_size,
                # shuffle=(train_sampler is None),
                shuffle=False,
                sampler=train_sampler,
                drop_last=True,
                pin_memory=True
            )  # num_workers=2
            return all_loader

        elif mode == 'contrast':
            contrast_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="contrast")

            if sampler:
                train_sampler = torch.utils.data.DistributedSampler(
                    contrast_dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True
                )
            else:
                train_sampler = None

            contrast_loader = DataLoader(
                dataset=contrast_dataset,
                batch_size=self.data_args.batch_size,
                shuffle=(train_sampler is None),
                drop_last=True,
                sampler=train_sampler,
                pin_memory=True
            )
            return contrast_loader
