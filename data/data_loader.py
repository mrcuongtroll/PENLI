import logging
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import pandas as pd
import os
from transformers import PreTrainedTokenizer, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
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
        :param model_type: (Type: int): 0: MLM, 1: Autoregressive Decoder, 2: Encoder-Decoder.
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
            raw_input = f"{hypothesis} {sep_token} {mask_token} . {premise}"
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
        elif self.model_type == 2:
            raw_inputs = f"Sentence 1: {premise}. Sentence 2: {hypothesis}. " \
                         f"Given sentence 1, is sentence 2 true, neutral, or false?"
            mapped_label = f"{GENERATIVE_LABEL_MAPPING[label]}"
            encodings = self.tokenizer.encode(raw_inputs)
            labels = self.tokenizer.encode(mapped_label)
            encodings = encodings[:min(len(encodings), self.max_seq_length)]
            return {'input_ids': encodings,
                    'labels': labels}
        else:
            raise RuntimeError(f"model_type 1 has not been implemented.")


# Functions
def create_nli_data_loader(tokenizer: PreTrainedTokenizer,
                           file_path: str = None,
                           max_seq_length: int = 512,
                           model_type: int = 0,
                           **kwargs):
    dataset = ESNLIDataset(tokenizer=tokenizer,
                           file_path=file_path,
                           max_seq_length=max_seq_length,
                           model_type=model_type)
    if model_type == 0:
        collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    elif model_type == 2:
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")
    data_loader = DataLoader(dataset, collate_fn=collator, **kwargs)
    return data_loader
