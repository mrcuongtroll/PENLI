import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForMaskedLM, T5ForConditionalGeneration
from transformers import BertTokenizer, T5Tokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings
import logging
from definitions import *
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class BertPENLI(nn.Module):

    def __init__(self,
                 pretrained: str = PLM[0],
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        """
        super(BertPENLI, self).__init__()
        # Load pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.bert = BertForMaskedLM.from_pretrained(pretrained)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        bert_rep = bert_output[0]
        output = self.softmax(bert_rep)
        return output

    def freeze_plm(self, freeze=True):
        for name, param in self.bert.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze


class T5PENLI(nn.Module):

    def __init__(self,
                 pretrained: str = PLM[2]
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        """
        super(T5PENLI, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained)
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        t5_outputs = self.t5(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             **kwargs)
        decoded_logits = t5_outputs.logits
        output = self.softmax(decoded_logits)
        return output

    def freeze_plm(self, freeze=True):
        for name, param in self.t5.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze

    def generate(self, **kwargs):
        return self.t5.generate(**kwargs)

    def save_plm(self, path):
        self.t5.save_pretrained(path)

    def load_plm(self, path):
        self.t5.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
