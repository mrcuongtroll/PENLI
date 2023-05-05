import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForMaskedLM, BertTokenizer
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

    def freeze_bert(self, freeze=True):
        for name, param in self.bert.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze
