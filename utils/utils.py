from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
import logging
from typing import Tuple, List
from model.model import *
from utils.config import ConfigParser
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


def compute_generative_results(predictions: List[np.ndarray],
                               ground_truth: List[np.ndarray]):
    """
    Compute generative encoder-decoder accuracy and natural language inference accuracy.
    :param predictions: list of predictions
    :param ground_truth: list of ground truth labels
    :return: {'inference_acc': ..., 'generation_acc': ...}
    """
    assert len(predictions) == len(ground_truth), "predictions and ground_truth must have matching length."
    num_tokens_total = 0
    num_tokens_correct = 0
    num_inference_correct = 0
    for i in range(len(predictions)):
        token_mask = ground_truth[i] != IGNORE_ID
        true_prediction_mask = (predictions[i] == ground_truth[i]) * token_mask
        num_tokens_correct += true_prediction_mask.sum()
        num_tokens_total += token_mask.sum()
        # Since we set the label to contain only 1 word, we assume that the first word is the answer we need
        if predictions[i][0] == ground_truth[i][0]:
            num_inference_correct += 1
    inference_acc = num_inference_correct / len(ground_truth)
    generative_acc = num_tokens_correct / num_tokens_total
    return {'inference_acc': inference_acc, 'generative_acc': generative_acc}


def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   criterion: Optimizer,
                   config: ConfigParser,
                   use_explanation: bool = False,
                   device: str = 'cuda'):
    model.to(device)
    model.eval()
    result = None
    loss_meter = AverageMeter()
    ground_truth = []
    predictions = []
    token_types = []
    num_batches_per_print = len(data_loader) // config['num_prints']
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(model, BertPENLI):
                assert data_loader.dataset.model_type == 0, "Set dataset's model_type to 0 when using Bert."
                input_ids, token_type_ids, attention_mask, labels = batch['input_ids'], \
                            batch['token_type_ids'], batch['attention_mask'], batch['labels']
                input_ids, token_type_ids, attention_mask, labels = (input_ids.to(device),
                                                                     token_type_ids.to(device),
                                                                     attention_mask.to(device),
                                                                     labels.to(device)
                                                                     )
                if use_explanation:
                    explanation, explanation_mask = batch['explanation'].to(device),\
                                                    batch['explanation_mask'].to(device)
                    input_ids = torch.cat([input_ids, explanation], dim=-1)
                    attention_mask = torch.cat([attention_mask, explanation_mask], dim=-1)
                    token_type_ids = torch.cat([token_type_ids, torch.ones(explanation.shape, dtype=torch.long)],
                                               dim=-1)
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
                token_types.extend(token_type_ids.detach().cpu().numpy())
            elif isinstance(model, T5PENLI):
                assert data_loader.dataset.model_type == 2, "Set dataset's model_type to 2 when using T5."
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                input_ids, attention_mask, labels = (input_ids.to(device),
                                                     attention_mask.to(device),
                                                     labels.to(device)
                                                     )
                if use_explanation:
                    explanation, explanation_mask = batch['explanation'].to(device),\
                                                    batch['explanation_mask'].to(device)
                    input_ids = torch.cat([input_ids, explanation], dim=-1)
                    attention_mask = torch.cat([attention_mask, explanation_mask], dim=-1)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            else:
                raise RuntimeError("Cannot match model type with dataset type.")
            p_outputs = torch.permute(outputs, (0, 2, 1))
            loss = criterion(p_outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            predictions.extend(pred.detach().cpu().numpy())
            ground_truth.extend(labels.detach().cpu().numpy())
            loss_meter.update(loss.item())
            result = {"loss": loss_meter.average()
                      }
            if (batch_idx + 1) % num_batches_per_print == 0:
                logger.info(f"Evaluating "
                            f"[{batch_idx * data_loader.batch_size}/{len(data_loader.dataset)} "
                            f"({(100. * batch_idx / len(data_loader)):.0f}%)] "
                            f"| Loss: {loss_meter.average():.5f} "
                            )
    if isinstance(model, BertPENLI):
        metrics = compute_masked_lm_results(predictions, ground_truth, token_types)
        result['inference_acc'] = metrics['inference_acc']
        result['mlm_acc'] = metrics['mlm_acc']
    elif isinstance(model, T5PENLI):
        metrics = compute_generative_results(predictions, ground_truth)
        result['inference_acc'] = metrics['inference_acc']
        result['generative_acc'] = metrics['generative_acc']
    return result
