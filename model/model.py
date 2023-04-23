import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForTokenClassification, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings
import logging
from langdetect import detect
from definitions import *
from typing import Tuple


logger = logging.getLogger(name=__name__)


# Classes
class PromptCLED(nn.Module):

    def __init__(self,
                 pretrained: str = 'bert-base-multilingual-cased',
                 num_labels: int = len(MINION_LABEL_TO_ID),
                 language_set: Tuple[str, ...] = MINION_LANGUAGE_CODES,
                 spec_prompt_size: int = 10,
                 gen_prompt_size: int = 10,
                 ):
        """
        :param pretrained: HuggingFace's pretrained model's name. Please refer to
                           (https://huggingface.co/transformers/v3.3.1/pretrained_models.html).
        :param num_labels: Number of event labels. (1 for 'O' and 2 for every other labels. i.e. B and I tags)
        :param language_set: A set of language codes for this cross-lingual model. Please refer to
                             https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for language codes.
        :param spec_prompt_size: Number of tokens for language specific prompt.
        :param gen_prompt_size: Number of tokens for general prompt.
        """
        super(PromptCLED, self).__init__()
        # Load pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.bert = BertModel.from_pretrained(pretrained)
        # Initialize prompts. They should be of shape (1, prompt_size, hidden_dim)
        self.num_languages = len(language_set)
        self.gen_prompt_size = gen_prompt_size
        self.general_prompt = nn.Parameter(data=torch.zeros((1, self.gen_prompt_size, self.bert.config.hidden_size)),
                                           requires_grad=True)
        nn.init.kaiming_normal_(self.general_prompt)
        self.spec_prompt_size = spec_prompt_size
        spec_prompt_list = {}
        for lang in language_set:
            spec_prompt = nn.Parameter(data=torch.zeros((1, self.spec_prompt_size, self.bert.config.hidden_size)),
                                       requires_grad=True)
            nn.init.kaiming_normal_(spec_prompt)
            spec_prompt_list[lang] = spec_prompt
        # spec_prompt_list['zh-cn'] = spec_prompt_list['ja']  # Because Japanese borrows lotsa characters from Chinese
        # spec_prompt_list['zh-tw'] = spec_prompt_list['ja']  # Same as above. Just in case...
        self.specific_prompts = nn.ParameterDict(spec_prompt_list)
        # Replace the embeddings layer with its custom wrapper
        self.bert.embeddings = BertEmbeddingsWrapper(self.bert.embeddings, general_prompt=self.general_prompt)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        specific_prompts = []
        for sentence in input_ids:
            raw_text = self.tokenizer.decode(sentence, skip_special_tokens=True)
            language = detect(raw_text)     # Not guarantee 100% accuracy
            if language in self.specific_prompts.keys():
                specific_prompts.append(self.specific_prompts[language])
            else:
                specific_prompts.append(self.specific_prompts['en'])    # TODO: find a better way to deal with this
        specific_prompts = torch.cat(specific_prompts, dim=0)
        # self.bert.embeddings.set_spec_prompt(self.specific_prompts['en'])
        self.bert.embeddings.set_spec_prompt(specific_prompts)
        bert_output = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        bert_rep = bert_output[0]
        output = self.classifier(bert_rep)
        output = self.softmax(output)
        return output

    def freeze_bert(self, freeze=True):
        for name, param in self.bert.named_parameters():
            if "prompt" not in name:
                param.requires_grad = not freeze


class BertEmbeddingsWrapper(nn.Module):

    def __init__(self, bert_embeddings: BertEmbeddings, general_prompt: nn.Parameter = None):
        super(BertEmbeddingsWrapper, self).__init__()
        self.bert_embeddings = bert_embeddings
        # Specific prompt should be of shape (batch_size, prompt_size, hidden_dim)
        self.language_specific_prompt = None
        # General prompt should be of shape (1, prompt_size, hidden_dim)
        self.general_prompt = general_prompt

    def forward(self, **kwargs):
        # Embeddings should be of shape (batch_size, sequence_length, hidden_dim)
        # Method #1: embed the input_ids without the reserved positions, then concat the prompts (to be considered)
        # Method #2: embed the input_ids with the reserved positions, then swap them with the prompts
        assert self.language_specific_prompt is not None, "Language specific prompt must be provided by calling " \
                                                          "BertEmbeddingsWrapper.set_spec_prompt(prompt)"
        bert_embeddings = self.bert_embeddings(**kwargs)
        if self.language_specific_prompt is not None:
            batch_size = bert_embeddings.shape[0]
            cls_tokens = bert_embeddings[:, 0:1, :]
            general_prompt = self.general_prompt.repeat(batch_size, 1, 1)
            prompt = torch.cat((general_prompt, self.language_specific_prompt), dim=-2)
            # prompt = prompt.repeat(batch_size, 1, 1)
            prompt_size = prompt.shape[-2]
            sentence_tokens = bert_embeddings[:, prompt_size + 1:, :]
            bert_embeddings = torch.cat((cls_tokens, prompt, sentence_tokens),
                                        dim=-2)
        return bert_embeddings

    def set_spec_prompt(self, prompt: nn.Parameter):
        """
        Set the language specific prompt to be concatenated into the embeddings.
        :param prompt: Language specific prompt.
        :return:
        """
        self.language_specific_prompt = prompt

    def set_gen_prompt(self, prompt: nn.Parameter):
        """
        Set the general CLED prompt to be concatenated into the embeddings.
        :param prompt: General prompt
        :return:
        """
        self.general_prompt = prompt


# TODO: docstrings
