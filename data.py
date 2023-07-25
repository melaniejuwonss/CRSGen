import argparse
from copy import deepcopy
from dataclasses import dataclass

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import torch
import json
import random

indexing_prefix = ["Review: ", "This is one of the review of item <blank>: ", "Review: ",
                   "This is one of the dialogs that describes the features of item <blank>: System: What did you think about the item <blank>? User: ",
                   "Here is the review: "]
indexing_postfix = ["Predict corresponding item: ", "Fill in the blank.", "Which item does it describe?",
                    "Fill in the blank.", "Pick the most suitable item from the following candidates: "]
recommend_prefix = ["Dialog: ", "This is one of the dialogs that ends up recommending item <blank>: ", "Dialog: ",
                    "This is a dialog that ends up recommending item <blank>: ",
                    "This is a dialog that ends up recommending item <blank>: "]
recommend_postfix = ["Predict next item: ", "Fill in the blank.", "Which item best matches with given dialog?",
                     "Fill in the blank.", "Fill in the blank."]


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            usePrefix: bool,
            usePostfix: bool,
            template_index: int,
            args: argparse.Namespace
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            verification_mode=False,
            cache_dir=cache_dir
        )['train']
        self.train_data = self.train_data.shuffle()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.usePrefix = usePrefix
        self.usePostfix = usePostfix
        self.template_index = template_index
        self.args = args
        self.all_item = json.load(open(f'data/Redial/{self.args.dataset}/item.json', 'r', encoding='utf-8'))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
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
            whole_text = whole_text.replace("Review: ", "")
            prefix = self.tokenizer(indexing_prefix[self.template_index], return_tensors="pt",
                                    add_special_tokens=False).input_ids
            prefix_length = prefix.size()[1]
            input_ids = self.tokenizer(whole_text,
                                       return_tensors="pt",
                                       padding="longest").input_ids
            input_ids = input_ids[:, :self.max_length]
            if self.usePostfix:
                if self.args.useCandidate:
                    tmp_item = deepcopy(self.all_item)
                    tmp_item.remove(data['item'])  # 정답은 마지막에 추가해줄거니까 random sample 후보에서 제거
                    random_sample_list = random.sample(tmp_item, self.args.cand_num)
                    random_sample_list.append(data['item'])
                    random.shuffle(random_sample_list)  # 순서 shuffle
                    sampled_item = ", ".join(random_sample_list)
                postfix = self.tokenizer(indexing_postfix[self.template_index] + sampled_item, return_tensors="pt",
                                         add_special_tokens=False).input_ids
                input_ids = torch.cat([prefix, input_ids, postfix], dim=1)[0]
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
            usePrefix: bool,
            usePostfix: bool,
            template_index: int,
            mode: str,
            args: argparse.Namespace
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            verification_mode=False,
            cache_dir=cache_dir
        )['train']
        if mode == "train":
            self.train_data = self.train_data.shuffle()

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.usePrefix = usePrefix
        self.usePostfix = usePostfix
        self.template_index = template_index
        self.args = args
        self.all_item = json.load(open(f'data/Redial/{self.args.dataset}/item.json', 'r', encoding='utf-8'))

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
                prefix = self.tokenizer(indexing_prefix[self.template_index], return_tensors="pt",
                                        add_special_tokens=False).input_ids
                prefix_length = prefix.size()[1]
                input_ids = self.tokenizer(whole_text,
                                           return_tensors="pt",
                                           padding="longest").input_ids
                input_ids = input_ids[:, :self.max_length]
                if self.usePostfix:
                    if self.args.useCandidate:
                        tmp_item = deepcopy(self.all_item)
                        tmp_item.remove(data['item'])  # 정답은 마지막에 추가해줄거니까 random sample 후보에서 제거
                        random_sample_list = random.sample(tmp_item, self.args.cand_num)
                        random_sample_list.append(data['item'])
                        random.shuffle(random_sample_list)  # 순서 shuffle
                        sampled_item = ", ".join(random_sample_list)
                    postfix = self.tokenizer(indexing_postfix[self.template_index] + sampled_item, return_tensors="pt",
                                             add_special_tokens=False).input_ids
                    input_ids = torch.cat([prefix, input_ids, postfix], dim=1)[0]
                else:
                    input_ids = input_ids[0]
            # elif whole_text.startswith('Movie information:'): # Meta 만 했을 경우
            #     input_ids = self.tokenizer(whole_text,
            #                                return_tensors="pt",
            #                                padding="longest").input_ids
            #     input_ids = input_ids[:, :self.max_length]
            #     postfix = self.tokenizer("Predict corresponding item: ", return_tensors="pt",
            #                              add_special_tokens=False).input_ids
            #     input_ids = torch.cat([input_ids, postfix], dim=1)[0]

            else:
                prefix = self.tokenizer(recommend_prefix[self.template_index],
                                        return_tensors="pt",
                                        add_special_tokens=False).input_ids
                prefix_length = prefix.size()[1]
                input_ids = self.tokenizer(whole_text,
                                           return_tensors="pt",
                                           padding="longest").input_ids
                input_ids = input_ids[:, -self.max_length:]
                if self.usePostfix:
                    postfix = self.tokenizer(recommend_postfix[self.template_index], return_tensors="pt",
                                             add_special_tokens=False).input_ids
                    input_ids = torch.cat([prefix, input_ids, postfix], dim=1)[0]
                else:
                    input_ids = input_ids[0]

        if self.usePrefix == False:
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
