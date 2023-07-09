import torch
import torch.nn as nn
from transformers import BertForMaskedLM, T5ForConditionalGeneration, GPT2LMHeadModel, AutoModelForMaskedLM
from transformers import BertTokenizer, T5Tokenizer, GPT2Tokenizer, AutoTokenizer
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
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.bert = AutoModelForMaskedLM.from_pretrained(pretrained)
        if hasattr(self.bert, 'roberta'):
            self.bert.roberta.config.type_vocab_size = 2  # BECAUSE REASONS
            token_type_embed = self.bert.roberta.embeddings.token_type_embeddings
            token_type_embed_weight = token_type_embed.weight.data
            token_type_embed = nn.Embedding(self.bert.roberta.config.type_vocab_size,
                                            self.bert.roberta.config.hidden_size)
            token_type_embed.weight.data = token_type_embed_weight.repeat(2, 1)
            self.bert.roberta.embeddings.token_type_embeddings = token_type_embed
        self.config = self.bert.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.bert.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def save_plm(self, path):
        self.bert.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.bert
        del self.config
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.bert = BertForMaskedLM.from_pretrained(path)
        self.config = self.bert.config


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
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained)
        self.config = self.t5.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.t5.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def generate(self, **kwargs):
        return self.t5.generate(**kwargs)

    def save_plm(self, path):
        self.t5.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.t5
        del self.config
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(path)
        self.config = self.t5.config


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
        self.gpt = GPT2LMHeadModel.from_pretrained(pretrained)
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        self.config = self.gpt.config
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.gpt(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels,
                           **kwargs)
        logits = outputs.logits
        softmaxed = self.softmax(logits)
        return softmaxed, outputs

    def freeze_plm(self, freeze=True):
        for name, param in self.gpt.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def generate(self, **kwargs):
        return self.gpt.generate(**kwargs)

    def save_plm(self, path):
        self.gpt.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_plm(self, path):
        del self.tokenizer
        del self.gpt
        del self.config
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.gpt = GPT2LMHeadModel.from_pretrained(path)
        self.config = self.gpt.config
