from prettytable import PrettyTable
import evaluate
import torch
import logging
import json
from definitions import *


logger = logging.getLogger(name=__name__)


# Objects
SEQEVAL = evaluate.load('seqeval')


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


def sequence_evaluation_metrics(predictions, labels):
    id2label = MINION_ID_TO_LABEL.copy()
    for idx in id2label.keys():
        label = id2label[idx]
        id2label[idx] = label.replace('-', '+').replace('_', '-').replace('+', '_')
    if isinstance(predictions, torch.Tensor):
        true_predictions = [
            [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    else:
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    if isinstance(labels, torch.Tensor):
        true_labels = [
            [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    else:
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    results = SEQEVAL.compute(predictions=true_predictions, references=true_labels)
    return results
