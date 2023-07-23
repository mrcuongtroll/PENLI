import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
import logging
from typing import Tuple, List
from model.model import *
from utils.config import ConfigParser
from utils.utils import AverageMeter
from definitions import *


logger = logging.getLogger(name=__name__)


# Functions:
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


def compute_autoregressive_results(tokenizer: PreTrainedTokenizerBase,
                                   predictions: List[np.ndarray],
                                   ground_truth: List[np.ndarray]):
    """
    Compute auto-regressive decoder-only accuracy and natural language inference accuracy.
    :param tokenizer: Tokenizer used to decode outputs.
    :param predictions: list of predictions
    :param ground_truth: list of ground truth labels
    :return: {'inference_acc': ..., 'generation_acc': ...}
    """
    assert len(predictions) == len(ground_truth), "predictions and ground_truth must have matching length."
    num_tokens_total = 0
    num_tokens_correct = 0
    num_inference_correct = 0
    # TODO: FIX THIS
    for i in range(len(predictions)):
        token_mask = ground_truth[i] != IGNORE_ID
        true_prediction_mask = (predictions[i] == ground_truth[i]) * token_mask
        num_tokens_correct += true_prediction_mask.sum()
        num_tokens_total += token_mask.sum()
        # Decode the prediction and labels to retrieve only the output part
        full_preds = [predictions[i][j] for j in range(len(ground_truth[i])) if ground_truth[i][j] != IGNORE_ID]
        full_labels = [ground_truth[i][j] for j in range(len(ground_truth[i])) if ground_truth[i][j] != IGNORE_ID]
        full_preds = tokenizer.decode(full_preds).split()
        full_labels = tokenizer.decode(full_labels).split()
        preds = None
        labels = None
        for p in full_preds:
            if p.startswith('Output: '):
                preds = p.replace('</s>', '').split()
        for l in full_labels:
            if l.startswith('Output: '):
                labels = l.replace('</s>', '').split()
        # Since we set the label to contain only 1 word, we assume that the first word is the answer we need
        if preds is not None and labels is not None and preds[1] == labels[1]:
            num_inference_correct += 1
    inference_acc = num_inference_correct / len(ground_truth)
    generative_acc = num_tokens_correct / num_tokens_total
    return {'inference_acc': inference_acc, 'generative_acc': generative_acc}


def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   criterion: nn.Module,
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
                outputs, full_outputs = model(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask,
                                              labels=labels)
                loss = full_outputs.loss
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
                outputs, full_outputs = model(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              labels=labels)
                loss = full_outputs.loss
            elif isinstance(model, GPTPENLI):
                assert data_loader.dataset.model_type == 1, "Set dataset's model_type to 2 when using GPT2."
                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                labels = batch['labels'].to(device)
                if use_explanation:
                    explanation, explanation_mask = batch['explanation'].to(device),\
                                                    batch['explanation_mask'].to(device)
                    input_ids = torch.cat([input_ids, explanation], dim=-1)
                    attention_mask = torch.cat([attention_mask, explanation_mask], dim=-1)
                outputs, full_outputs = model(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              labels=labels)
                loss = full_outputs.loss
            else:
                raise RuntimeError("Cannot match model type with dataset type.")
            # p_outputs = torch.permute(outputs, (0, 2, 1))
            # loss = criterion(p_outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            predictions.extend(pred.detach().cpu().numpy())
            ground_truth.extend(labels.detach().cpu().numpy())
            loss_meter.update(loss.item())
            result = {"loss": loss_meter.average()}
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
    elif isinstance(model, GPTPENLI):
        metrics = compute_autoregressive_results(model.tokenizer, predictions, ground_truth)
        result['inference_acc'] = metrics['inference_acc']
        result['generative_acc'] = metrics['generative_acc']
    return result


def evaluate_baseline(model: nn.Module,
                      data_loader: DataLoader,
                      criterion: nn.Module,
                      config: ConfigParser,
                      device: str = 'cuda'):
    model.to(device)
    model.eval()
    result = None
    loss_meter = AverageMeter()
    ground_truth = []
    predictions = []
    num_batches_per_print = len(data_loader) // config['num_prints']
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(model, MLMBaseline):
                assert data_loader.dataset.model_type == 0, "Set dataset's model_type to 0 when using Bert."
                input_ids, token_type_ids, attention_mask, labels = batch['input_ids'], \
                                                                    batch['token_type_ids'], batch['attention_mask'], \
                                                                    batch['labels']
                input_ids, token_type_ids, attention_mask = (input_ids.to(device),
                                                             token_type_ids.to(device),
                                                             attention_mask.to(device)
                                                             )
                labels = labels.to(device)
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            else:
                raise RuntimeError("Cannot match model type with dataset type.")
            loss = criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            predictions.extend(pred.detach().cpu().numpy())
            ground_truth.extend(labels.detach().cpu().numpy())
            loss_meter.update(loss.item())
            result = {"loss": loss_meter.average()}
            if (batch_idx + 1) % num_batches_per_print == 0:
                logger.info(f"Evaluating "
                            f"[{batch_idx * data_loader.batch_size}/{len(data_loader.dataset)} "
                            f"({(100. * batch_idx / len(data_loader)):.0f}%)] "
                            f"| Loss: {loss_meter.average():.5f} "
                            )
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    acc = (ground_truth == predictions).sum() / len(data_loader.dataset)
    result['acc'] = acc
    return result
