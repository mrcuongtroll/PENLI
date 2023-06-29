import torch
import numpy as np
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class GeneralDataCollator:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        return_dict = {}
        keys = features[0].keys()
        assert "labels" in keys, "There must be one feature named 'labels'."
        for k in keys:
            values = [feature[k] for feature in features]
            return_dict[k] = self._make_padded_tensor(values, k, return_tensors)
            # Hard-code this part
            if k == 'input_ids':
                return_dict['attention_mask'] = (return_dict[k] != self.tokenizer.pad_token_id).long()
            elif k == 'explanation':
                return_dict['explanation_mask'] = (return_dict[k] != self.tokenizer.pad_token_id).long()
        return return_dict

    def _make_padded_tensor(self, item_list: Union[list, tuple], key: str = None, return_tensors: str = 'pt'):
        max_item_len = 0
        for item in item_list:
            if len(item) > max_item_len:
                max_item_len = len(item)
        for item in item_list:
            padding = self.label_pad_token_id if key == 'labels' else self.tokenizer.pad_token_id
            remainder = [padding] * (max_item_len - len(item))
            item.extend(remainder)
        if return_tensors == 'pt':
            return torch.tensor(item_list, dtype=torch.long)
