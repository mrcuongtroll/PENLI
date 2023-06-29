import torch
import torch.nn as nn
from transformers import BertForMaskedLM, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers import BertTokenizer, T5Tokenizer, GPT2Tokenizer
import logging
from definitions import *
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class BertPENLI(nn.Module):

    def __init__(self,
                 pretrained: str = PLM[0]
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        """
        super(BertPENLI, self).__init__()
        # Load pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertForMaskedLM.from_pretrained(pretrained)
        self.config = self.model.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.model.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def save_plm(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.model
        del self.config
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForMaskedLM.from_pretrained(path)
        self.config = self.model.config


class T5PENLI(nn.Module):

    def __init__(self,
                 pretrained: str = PLM[2]
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        """
        super(T5PENLI, self).__init__()
        # Load pretrained model
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained)
        self.config = self.model.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.model.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_plm(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.model
        del self.config
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.config = self.model.config


class GPTPENLI(nn.Module):

    def __init__(self,
                 pretrained: str = PLM[1]
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        """
        super(GPTPENLI, self).__init__()
        # Load pretrained model
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.model = GPT2LMHeadModel.from_pretrained(pretrained)
        self.config = self.model.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.model.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_plm(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.model
        del self.config
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.config = self.model.config
