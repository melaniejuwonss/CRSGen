from dataclasses import dataclass

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import torch


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            usePrefix: bool
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            verification_mode=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.usePrefix = usePrefix

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        prefix_length = 0
        data = self.train_data[item]

        text = data['context_tokens']

        self.tokenizer.padding_side = "left"
        if self.usePrefix:
            prefix = self.tokenizer("Review: ", return_tensors="pt").input_ids
            prefix_length = prefix.size()[1]

        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="longest").input_ids

        input_ids = input_ids[:, -self.max_length + prefix_length:]
        if self.usePrefix:
            input_ids = torch.cat([prefix, input_ids], dim=1)[0]
        else:
            input_ids = input_ids[0]

        return input_ids, str(data['item'])


class RecommendTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            usePrefix: bool
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            verification_mode=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.usePrefix = usePrefix

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        prefix_length = 0
        data = self.train_data[item]

        text = data['context_tokens']
        whole_text = ""
        for idx, sentence in enumerate(text):
            if idx == (len(text) - 1):
                whole_text += sentence
            else:
                whole_text += (sentence + " " + self.tokenizer.eos_token + " ")

        # input_ids = self.tokenizer(whole_text,
        #                            return_tensors="pt",
        #                            truncation='only_first',
        #                            max_length=self.max_length,
        #                            padding="longest").input_ids[0]
        self.tokenizer.padding_side = "left"
        if self.usePrefix:
            if whole_text.startswith('Review:'):
                whole_text = whole_text.replace("Review: ", "")
                prefix = self.tokenizer("Review: ", return_tensors="pt", add_special_tokens=False).input_ids
            else:
                prefix = self.tokenizer("Dialog: ", return_tensors="pt", add_special_tokens=False).input_ids
            prefix_length = prefix.size()[1]
        input_ids = self.tokenizer(whole_text,
                                   return_tensors="pt",
                                   padding="longest").input_ids

        input_ids = input_ids[:, -self.max_length + prefix_length:]
        if self.usePrefix:
            input_ids = torch.cat([prefix, input_ids], dim=1)[0]
        else:
            input_ids = input_ids[0]

        return input_ids, str(data['item'])


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels
