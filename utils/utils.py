from prettytable import PrettyTable
import torch
import numpy as np
import logging
import json
from typing import Tuple, List
from definitions import *


logger = logging.getLogger(name=__name__)


# Classes
class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count += weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)

        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


# Functions
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def compute_masked_lm_results(predictions: List[np.ndarray],
                              ground_truth: List[np.ndarray],
                              token_type_ids: List[np.ndarray]):
    """
    Compute masked language modeling accuracy and natural language inference accuracy.
    :param predictions: list of predictions
    :param ground_truth: list of ground truth labels
    :param token_type_ids: the token_type_ids corresponding to the samples. (0 for the first sentence, 1 for the second)
    :return: {'inference_acc': ..., 'mlm_acc': ...}
    """
    assert len(predictions) == len(ground_truth), "predictions and ground_truth must have matching length."
    assert len(token_type_ids) == len(ground_truth), "token_type_ids and ground_truth must have matching length"
    num_tokens_total = 0
    num_tokens_correct = 0
    num_inference_correct = 0
    for i in range(len(predictions)):
        token_mask = ground_truth[i] != IGNORE_ID
        true_prediction_mask = (predictions[i] == ground_truth[i]) * token_mask
        num_tokens_correct += true_prediction_mask.sum()
        num_tokens_total += token_mask.sum()
        for j in range(len(token_type_ids[i])):
            if token_type_ids[i][j] == 1:
                if predictions[i][j] == ground_truth[i][j]:
                    num_inference_correct += 1
                break
    inference_acc = num_inference_correct / len(ground_truth)
    mlm_acc = num_tokens_correct / num_tokens_total
    return {'inference_acc': inference_acc, 'mlm_acc': mlm_acc}
