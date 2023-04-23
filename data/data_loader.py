import logging
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import os
from transformers import PreTrainedTokenizer, DataCollatorForTokenClassification
from definitions import *
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class MINIONDataset(Dataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str = None,
                 languages: Tuple[str, ...] = None,
                 dataset_type: str = None,
                 max_seq_length: int = 512,
                 prompt_reserve_size: int = 20,
                 subword=True,
                 ignore_o=False,
                 trigger_sampling_weight: int = 5.0
                 ):
        """
        :param file_path: (Type: str) Path to the file containing the data. The file should consists of lines of json
                          strings containing the tokens and the labels. If this is provided then ignore languages and
                          dataset_type.
        :param tokenizer: (Type: PreTrainedTokenizer) The tokenizer object used to encode the data.
        :param languages: (Type: Tuple[str, ...]) A list of languages to include in the dataset. All of these languages must
                          be available in the MINION dataset.
        :param dataset_type: (Type: str) 'train', 'valid' or 'test'.
        :param max_seq_length: (Type: int) The maximum number of tokens for each sentence.
        :param subword: (Type: boolean) Whether to use sub-word tokenization or not. It is advised to set this to True,
                        though it is significantly slower to load the data.
        :param prompt_reserve_size: (Type: int) Number of tokens to reserve for prompts. These will be treated as mask
                                    tokens. This number should fit the prompt size of the model. i.e. the total size of
                                    general prompt and language specific prompt.
        :param ignore_o: (Type: boolean) Whether to use 'O' tokens to calculate the loss or not.
        """
        super(MINIONDataset, self).__init__()
        if file_path is not None:
            logger.info("Since file_path is provided, languages and dataset_type are ignored.")
            with open(file_path, 'r', encoding='utf8') as f:
                self.data = f.readlines()
        else:
            assert languages is not None and dataset_type is not None, "languages and dataset_type must be provided " \
                                                                       "when file_path is not available."
            for lang in languages:
                assert lang in MINION_LANGUAGES, f"The provided language {lang} is not available in the MINION " \
                                                 f"dataset {MINION_LANGUAGES}."
            assert dataset_type in DATASET_TYPES, f"Invalid dataset type. Choose one from {DATASET_TYPES}."
            self.data = []
            for lang in languages:
                lang_path = os.path.join(MINION_DIR, lang, f'{dataset_type}.json')
                with open(lang_path, 'r', encoding='utf8') as f:
                    self.data.extend(f.readlines())
        self.len = len(self.data)
        self.tokenizer = tokenizer
        self.subword = subword
        self.prompt_reserve_size = prompt_reserve_size
        self.max_seq_length = max_seq_length
        self.ignore_o = ignore_o
        self.trigger_sampling_weight = trigger_sampling_weight

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.subword:
            # Apparently this one should never be used (cuz it's wrong)
            data = json.loads(self.data[idx])
            # Prepend a '[CLS]' token and append a '[SEP]' token, then convert to tokenized ids.
            encodings = self.tokenizer.encode(data['tokens'])
            labels = [MINION_LABEL_TO_ID[label] for label in data['labels']]
            # Treat [CLS] and [SEP] as ignored tokens
            labels = [-100] + labels + [-100]
            # Treat [CLS] and [SEP] as 'O' tokens
            # labels = [0] + labels + [0]
            return {'input_ids': encodings, 'labels': labels}
        else:
            data = json.loads(self.data[idx])
            tokens = data['tokens']
            raw_labels = data['labels']
            encodings = []
            labels = []
            for i in range(len(tokens)):
                encoded_tokens = self.tokenizer.encode(tokens[i], add_special_tokens=False)
                encodings += encoded_tokens
                if raw_labels[i].startswith("B"):
                    # The first sub-word is still B token
                    labels += [MINION_LABEL_TO_ID[raw_labels[i]]]
                    # The rest of the word are treated as I tokens
                    labels += [MINION_LABEL_TO_ID[raw_labels[i]] + 1] * (len(encoded_tokens) - 1)
                elif raw_labels[i].startswith("I"):
                    labels += [MINION_LABEL_TO_ID[raw_labels[i]]] * len(encoded_tokens)
                elif raw_labels[i].startswith("O"):
                    if self.ignore_o:
                        labels += [-100] * len(encoded_tokens)
                    else:
                        labels += [MINION_LABEL_TO_ID[raw_labels[i]]] * len(encoded_tokens)
            # Add special tokens: [CLS]: 101, [SEP]: 102, [MASK]: 103
            encodings = [101] + ([103] * self.prompt_reserve_size) + [102] + encodings + [102]
            # Treat [CLS] and [SEP] as ignored tokens
            labels = [-100] + ([-100] * self.prompt_reserve_size) + [-100] + labels + [-100]
            # Treat [CLS] and [SEP] as 'O' tokens
            # labels = [0] + ([0] * self.prompt_reserve_size) + [0] + labels + [0]
            first_seq_len = 1 + self.prompt_reserve_size + 1
            second_seq_len = len(labels) - first_seq_len
            token_type_ids = ([0] * first_seq_len) + ([1] * second_seq_len)
            # Truncate sequence to 512 tokens maximum (BERT limit)
            encodings = encodings[:min(len(encodings), 512)]
            token_type_ids = token_type_ids[:min(len(token_type_ids), 512)]
            labels = labels[:min(len(labels), self.max_seq_length)]
            return {'input_ids': encodings,
                    'token_type_ids': token_type_ids,
                    'labels': labels}

    def get_sampling_weight(self):
        sampling_weight = []
        for raw_data in self.data:
            data = json.loads(raw_data)
            labels = data['labels']
            contains_trigger = False
            for label in labels:
                if label != "O":
                    contains_trigger = True
                    break
            if contains_trigger:
                sampling_weight.append(self.trigger_sampling_weight)
            else:
                sampling_weight.append(1.0)
        return sampling_weight


# Functions
def token_classification_data_loader(tokenizer: PreTrainedTokenizer,
                                     file_path: str = None,
                                     languages: Tuple[str, ...] = None,
                                     dataset_type: str = None,
                                     max_seq_length: int = 512,
                                     prompt_reserve_size: int = 20,
                                     subword=True,
                                     ignore_o=False,
                                     trigger_sampling_weight: int = 5.0,
                                     **kwargs):
    dataset = MINIONDataset(tokenizer=tokenizer,
                            file_path=file_path,
                            languages=languages,
                            dataset_type=dataset_type,
                            max_seq_length=max_seq_length,
                            prompt_reserve_size=prompt_reserve_size,
                            subword=subword,
                            ignore_o=ignore_o,
                            trigger_sampling_weight=trigger_sampling_weight)
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    sampling_weight = dataset.get_sampling_weight()
    sampler = None
    if dataset_type in ("train", "training") or (file_path is not None and "train" in file_path):
        sampler = WeightedRandomSampler(sampling_weight, len(sampling_weight))
    data_loader = DataLoader(dataset, collate_fn=collator, sampler=sampler, **kwargs)
    return data_loader
