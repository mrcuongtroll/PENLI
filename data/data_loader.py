import logging
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import json
import os
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from .data_collator import GeneralDataCollator
from definitions import *
from utils.utils import most_frequent
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class ESNLIDataset(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 max_seq_length: int = 512,
                 model_type: int = 0,
                 use_explanation: bool = False
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. It must be in csv format.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param model_type: (Type: int): 0: MLM, 1: Autoregressive Decoder, 2: Encoder-Decoder.
        :param use_explanation: (Type: bool) Whether to return explanations or not.
        """
        super(ESNLIDataset, self).__init__()
        self.data = pd.read_csv(file_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        assert model_type in (0, 1, 2), f"Invalid model type: {model_type}. Please choose from " \
                                        f"(0: Masked Language Model, 1: Autoregressive Decoder, 2: Encoder-Decoder)."
        self.model_type = model_type
        self.use_explanation = use_explanation

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        premise = str(data['Sentence1'])
        hypothesis = str(data['Sentence2'])
        label = str(data['gold_label'])
        explanation = str(data['Explanation_1'])
        if self.model_type == 0:
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            sep_token = self.tokenizer.sep_token
            mask_token_id = self.tokenizer.mask_token_id
            ignored_token_id = IGNORE_ID
            mask_token = self.tokenizer.mask_token
            token_type = 0
            token_type_ids = []
            labels = []
            raw_input = f"{hypothesis}{sep_token} {mask_token} . {premise}"
            encodings = self.tokenizer.encode(raw_input)
            for encoding in encodings:
                token_type_ids.append(token_type)
                if encoding == sep_token_id:
                    token_type = 1
            # Special tokens: [CLS]: 101, [SEP]: 102, [MASK]: 103
            # raw_labels = f"{hypothesis} {sep_token} {LABEL_MAPPING[label]} . {premise}"
            # labels = self.tokenizer.encode(raw_labels, do_lower_case=True)
            labels.extend(self.tokenizer.encode(hypothesis, add_special_tokens=False))
            labels.append(ignored_token_id)
            labels.extend(self.tokenizer.encode(f"{MLM_LABEL_MAPPING[label]} . {premise}",
                                                add_special_tokens=False
                                                )
                          )
            labels = [ignored_token_id] + labels + [ignored_token_id]

            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            labels = labels[:min(len(labels), self.max_seq_length)]
            token_type_ids = token_type_ids[:min(len(token_type_ids), self.max_seq_length)]
            if not self.use_explanation:
                return {'input_ids': encodings,
                        'token_type_ids': token_type_ids,
                        'labels': labels}
            else:
                encoded_explanation = self.tokenizer.encode(explanation)
                encoded_explanation = encoded_explanation[:min(len(encoded_explanation), self.max_seq_length)]
                return {'input_ids': encodings,
                        'token_type_ids': token_type_ids,
                        'explanation': encoded_explanation,
                        'labels': labels}
        elif self.model_type == 1:
            raw_inputs = f"Instruction: Given sentence 1, is sentence 2 true, neutral, or false?\n" \
                         f"Sentence 1: {premise}\n" \
                         f"Sentence 2: {hypothesis}\n" \
                         f"Output: {GENERATIVE_LABEL_MAPPING[label]}"
            encodings = self.tokenizer.encode(raw_inputs)
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            if not self.use_explanation:
                return {'input_ids': encodings,
                        'labels': encodings}
            else:
                encoded_explanation = self.tokenizer.encode(explanation)
                encoded_explanation = encoded_explanation[:min(len(encoded_explanation), self.max_seq_length)]
                return {'input_ids': encodings,
                        'explanation': encoded_explanation,
                        'labels': encodings}
        elif self.model_type == 2:
            raw_inputs = f"Sentence 1: {premise}. Sentence 2: {hypothesis}. " \
                         f"Given sentence 1, is sentence 2 true, neutral, or false?"
            encodings = self.tokenizer.encode(raw_inputs)
            mapped_label = f"{GENERATIVE_LABEL_MAPPING[label]}"
            labels = self.tokenizer.encode(mapped_label)
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            labels = labels[:min(len(labels), self.max_seq_length)]
            if not self.use_explanation:
                return {'input_ids': encodings,
                        'labels': labels}
            else:
                encoded_explanation = self.tokenizer.encode(explanation)
                encoded_explanation = encoded_explanation[:min(len(encoded_explanation), self.max_seq_length)]
                return {'input_ids': encodings,
                        'explanation': encoded_explanation,
                        'labels': labels}


class ESNLIDatasetForBaseline(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 max_seq_length: int = 512,
                 model_type: int = 0
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. It must be in csv format.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param model_type: (Type: int): 0: MLM, 1: Autoregressive Decoder, 2: Encoder-Decoder.
        """
        super(ESNLIDatasetForBaseline, self).__init__()
        self.data = pd.read_csv(file_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        assert model_type in (0, 1, 2), f"Invalid model type: {model_type}. Please choose from " \
                                        f"(0: Masked Language Model, 1: Autoregressive Decoder, 2: Encoder-Decoder)."
        self.model_type = model_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        premise = str(data['Sentence1'])
        hypothesis = str(data['Sentence2'])
        label = str(data['gold_label'])
        explanation = str(data['Explanation_1'])
        label = CLASSIFICATION_LABEL_MAPPING[label]
        if self.model_type == 0:
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            sep_token = self.tokenizer.sep_token
            mask_token_id = self.tokenizer.mask_token_id
            ignored_token_id = IGNORE_ID
            mask_token = self.tokenizer.mask_token
            token_type = 0
            token_type_ids = []
            raw_input = f"{hypothesis}{sep_token}{premise}"
            encodings = self.tokenizer.encode(raw_input)
            for encoding in encodings:
                token_type_ids.append(token_type)
                if encoding == sep_token_id:
                    token_type = 1
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            token_type_ids = token_type_ids[:min(len(token_type_ids), self.max_seq_length)]
            return {'input_ids': encodings,
                    'token_type_ids': token_type_ids,
                    'label': label}
        elif self.model_type == 1:
            raise RuntimeError("model_type=1 has not been implemented for baseline datasets")
        elif self.model_type == 2:
            encodings = None
            return {'input_ids': encodings,
                    'token_type_ids': token_type_ids,
                    'label': label}


class MNLIDataset(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 max_seq_length: int = 512,
                 model_type: int = 0
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. It must be in json or jsonl format.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param model_type: (Type: int): 0: MLM, 1: Autoregressive Decoder, 2: Encoder-Decoder.
        """
        super(MNLIDataset, self).__init__()
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf8') as f:
            self.data = f.readlines()
        self.len = len(self.data)
        self.max_seq_length = max_seq_length
        assert model_type in (0, 1, 2), f"Invalid model type: {model_type}. Please choose from " \
                                        f"(0: Masked Language Model, 1: Autoregressive Decoder, 2: Encoder-Decoder)."
        self.model_type = model_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = json.loads(self.data[idx])
        premise = str(data['sentence1'])
        hypothesis = str(data['sentence2'])
        label = str(data['gold_label'])
        if label not in GENERATIVE_LABEL_MAPPING:
            label = most_frequent(list(map(str, data['annotator_labels'])))
        if self.model_type == 0:
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            sep_token = self.tokenizer.sep_token
            mask_token_id = self.tokenizer.mask_token_id
            ignored_token_id = IGNORE_ID
            mask_token = self.tokenizer.mask_token
            token_type = 0
            token_type_ids = []
            labels = []
            raw_input = f"{hypothesis}{sep_token} {mask_token} . {premise}"
            encodings = self.tokenizer.encode(raw_input)
            for encoding in encodings:
                token_type_ids.append(token_type)
                if encoding == sep_token_id:
                    token_type = 1
            # Special tokens: [CLS]: 101, [SEP]: 102, [MASK]: 103
            # raw_labels = f"{hypothesis} {sep_token} {LABEL_MAPPING[label]} . {premise}"
            # labels = self.tokenizer.encode(raw_labels, do_lower_case=True)
            labels.extend(self.tokenizer.encode(hypothesis, add_special_tokens=False))
            labels.append(ignored_token_id)
            labels.extend(self.tokenizer.encode(f"{MLM_LABEL_MAPPING[label]} . {premise}",
                                                add_special_tokens=False
                                                )
                          )
            labels = [ignored_token_id] + labels + [ignored_token_id]
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            labels = labels[:min(len(labels), self.max_seq_length)]
            token_type_ids = token_type_ids[:min(len(token_type_ids), self.max_seq_length)]
            return {'input_ids': encodings,
                    'token_type_ids': token_type_ids,
                    'labels': labels}
        elif self.model_type == 1:
            raw_inputs = f"Instruction: Given sentence 1, is sentence 2 true, neutral, or false?\n" \
                         f"Sentence 1: {premise}\n" \
                         f"Sentence 2: {hypothesis}\n" \
                         f"Output: {GENERATIVE_LABEL_MAPPING[label]}"
            encodings = self.tokenizer.encode(raw_inputs)
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            return {'input_ids': encodings,
                    'labels': encodings}
        elif self.model_type == 2:
            raw_inputs = f"Sentence 1: {premise}. Sentence 2: {hypothesis}. " \
                         f"Given sentence 1, is sentence 2 true, neutral, or false?"
            encodings = self.tokenizer.encode(raw_inputs)
            mapped_label = f"{GENERATIVE_LABEL_MAPPING[label]}"
            labels = self.tokenizer.encode(mapped_label)
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            labels = labels[:min(len(labels), self.max_seq_length)]
            return {'input_ids': encodings,
                    'labels': labels}


class MNLIDatasetForBaseline(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 max_seq_length: int = 512,
                 model_type: int = 0
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. It must be in json or jsonl format.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param model_type: (Type: int): 0: MLM, 1: Autoregressive Decoder, 2: Encoder-Decoder.
        """
        super(MNLIDatasetForBaseline, self).__init__()
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf8') as f:
            self.data = f.readlines()
        self.len = len(self.data)
        self.max_seq_length = max_seq_length
        assert model_type in (0, 1, 2), f"Invalid model type: {model_type}. Please choose from " \
                                        f"(0: Masked Language Model, 1: Autoregressive Decoder, 2: Encoder-Decoder)."
        self.model_type = model_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = json.loads(self.data[idx])
        premise = str(data['sentence1'])
        hypothesis = str(data['sentence2'])
        label = str(data['gold_label'])
        if label not in CLASSIFICATION_LABEL_MAPPING:
            label = most_frequent(list(map(str, data['annotator_labels'])))
        label = CLASSIFICATION_LABEL_MAPPING[label]
        if self.model_type == 0:
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            sep_token = self.tokenizer.sep_token
            mask_token_id = self.tokenizer.mask_token_id
            ignored_token_id = IGNORE_ID
            mask_token = self.tokenizer.mask_token
            token_type = 0
            token_type_ids = []
            raw_input = f"{hypothesis}{sep_token}{premise}"
            encodings = self.tokenizer.encode(raw_input)
            for encoding in encodings:
                token_type_ids.append(token_type)
                if encoding == sep_token_id:
                    token_type = 1
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            token_type_ids = token_type_ids[:min(len(token_type_ids), self.max_seq_length)]
            return {'input_ids': encodings,
                    'token_type_ids': token_type_ids,
                    'label': label}
        elif self.model_type == 1:
            raise RuntimeError("model_type=1 has not been implemented for baseline datasets")
        elif self.model_type == 2:
            raw_inputs = f"Sentence 1: {premise}. Sentence 2: {hypothesis}. " \
                         f"Given sentence 1, is sentence 2 true, neutral, or false?"
            encodings = self.tokenizer.encode(raw_inputs)
            mapped_label = f"{GENERATIVE_LABEL_MAPPING[label]}"
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            return {'input_ids': encodings,
                    'label': label}


class ResumableRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    # data_source: Sized
    # replacement: bool

    def __init__(self, data_source, seed=69420):
        super(ResumableRandomSampler, self).__init__(data_source=data_source)
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1].item()

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state):
        self.perm = torch.tensor(state["perm"], dtype=torch.int)
        self.perm_index = state["perm_index"]
        self.generator.set_state(torch.ByteTensor(state["generator_state"].to('cpu')))


# Functions
def create_nli_data_loader(tokenizer: PreTrainedTokenizer,
                           file_path: str = None,
                           max_seq_length: int = 512,
                           model_type: int = 0,
                           use_explanation: bool = False,
                           seed=69420,
                           **kwargs):
    if 'esnli' in file_path:
        dataset = ESNLIDataset(tokenizer=tokenizer,
                               file_path=file_path,
                               max_seq_length=max_seq_length,
                               model_type=model_type,
                               use_explanation=use_explanation)
    elif 'multinli' in file_path:
        dataset = MNLIDataset(tokenizer=tokenizer,
                              file_path=file_path,
                              max_seq_length=max_seq_length,
                              model_type=model_type)
    # LEGACY CODE
    # if model_type == 0:
    #     collator = DataCollatorForTokenClassification(tokenizer, padding="max_length")
    # elif model_type == 2:
    #     collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    # else:
    #     raise RuntimeError(f"Invalid model type: {model_type}")
    # LEGACY CODE
    collator = GeneralDataCollator(tokenizer=tokenizer)
    sampler = ResumableRandomSampler(data_source=dataset, seed=seed)
    data_loader = DataLoader(dataset, collate_fn=collator, sampler=sampler, **kwargs)
    return data_loader


def create_nli_data_loader_for_baseline(tokenizer: PreTrainedTokenizer,
                                        file_path: str = None,
                                        max_seq_length: int = 512,
                                        model_type: int = 0,
                                        seed=69420,
                                        **kwargs):
    if 'esnli' in file_path:
        dataset = ESNLIDatasetForBaseline(tokenizer=tokenizer,
                                          file_path=file_path,
                                          max_seq_length=max_seq_length,
                                          model_type=model_type)
    elif 'multinli' in file_path:
        dataset = MNLIDatasetForBaseline(tokenizer=tokenizer,
                                         file_path=file_path,
                                         max_seq_length=max_seq_length,
                                         model_type=model_type)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    sampler = ResumableRandomSampler(data_source=dataset, seed=seed)
    data_loader = DataLoader(dataset, collate_fn=collator, sampler=sampler, **kwargs)
    return data_loader
