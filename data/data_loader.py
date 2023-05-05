import logging
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import pandas as pd
import os
from transformers import PreTrainedTokenizer, DataCollatorForTokenClassification
from definitions import *
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class ESNLIDataset(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 max_seq_length: int = 512,
                 model_type: int = 0,
                 prompt_reserve_size: int = 20
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. It must be in csv format.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param model_type: (Type: int): 0: MLM, 1: Generative decoder, 2: encoder-decoder.
        :param prompt_reserve_size: (Type: int) Number of tokens to reserve for prompts. These will be treated as mask
                                    tokens. This number should fit the prompt size of the model. i.e. the total size of
                                    general prompt and language specific prompt.
        """
        super(ESNLIDataset, self).__init__()
        self.data = pd.read_csv(file_path)
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.prompt_reserve_size = prompt_reserve_size
        self.max_seq_length = max_seq_length
        self.model_type = model_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        sep_token = self.tokenizer.sep_token
        mask_token_id = self.tokenizer.mask_token_id
        ignored_token_id = IGNORE_ID
        mask_token = self.tokenizer.mask_token
        data = self.data.iloc[idx]
        premise = data['Sentence1']
        hypothesis = data['Sentence2']
        label = data['gold_label']
        explanation = data['Explanation_1']
        token_type = 0
        token_type_ids = []
        labels = []
        if self.model_type == 0:
            raw_input = f"{hypothesis} {sep_token} {mask_token} . {premise}"
            encodings = self.tokenizer.encode(raw_input, do_lower_case=True)
            for encoding in encodings:
                token_type_ids.append(token_type)
                if encoding == sep_token_id:
                    token_type = 1
            # Special tokens: [CLS]: 101, [SEP]: 102, [MASK]: 103
            # raw_labels = f"{hypothesis} {sep_token} {LABEL_MAPPING[label]} . {premise}"
            # labels = self.tokenizer.encode(raw_labels, do_lower_case=True)
            labels.extend(self.tokenizer.encode(hypothesis, add_special_tokens=False))
            labels.append(ignored_token_id)
            labels.extend(self.tokenizer.encode(f"{LABEL_MAPPING[label]} . {premise}",
                                                add_special_tokens=False
                                                )
                          )
            labels = [ignored_token_id] + labels + [ignored_token_id]
        else:
            raise RuntimeError(f"model_type 1 and 2 have not been implemented.")
        encodings = encodings[:min(len(encodings), self.max_seq_length)]
        labels = labels[:min(len(labels), self.max_seq_length)]
        token_type_ids = token_type_ids[:min(len(token_type_ids), self.max_seq_length)]
        return {'input_ids': encodings,
                'token_type_ids': token_type_ids,
                'labels': labels}


# Functions
def token_classification_data_loader(tokenizer: PreTrainedTokenizer,
                                     file_path: str = None,
                                     max_seq_length: int = 512,
                                     **kwargs):
    dataset = ESNLIDataset(tokenizer=tokenizer,
                           file_path=file_path,
                           max_seq_length=max_seq_length,
                           model_type=0)
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    data_loader = DataLoader(dataset, collate_fn=collator, **kwargs)
    return data_loader
