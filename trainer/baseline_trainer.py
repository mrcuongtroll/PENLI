import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from utils.utils import AverageMeter
from utils.config import ConfigParser
from model.model import MLMBaseline
from definitions import *

logger = logging.getLogger(name=__name__)


# Classes
class BaselineTrainer:

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 valid_loader: DataLoader = None,
                 lr_scheduler=None,
                 config: ConfigParser = None,
                 device='cuda'
                 ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.current_epoch = 0
        self.best_acc = 0
        if self.config['freeze_plm']:
            self.model.freeze_plm(freeze=True)

    def train(self, resume=False):
        with torch.autograd.set_detect_anomaly(True):
            if resume:
                self._load_model()
                logger.info(f"------> Resume training from epoch {self.current_epoch}...")
            else:
                logger.info(f"------> Begin training...")
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                logger.info(f"------> Current epoch: {epoch}")
                train_result = self._train_epoch(epoch)

                if self.valid_loader is not None:
                    valid_result = self._eval_epoch(epoch)
                    valid_loss = valid_result['loss']
                    valid_acc = valid_result['acc']
                    valid_result[
                        'state_dict'] = self.model.state_dict() if self.model else self.model.state_dict()
                    valid_result['optimizer'] = self.optimizer.state_dict()
                    valid_result['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
                    valid_result['epoch'] = epoch + 1
                    valid_result['train_sampler'] = self.train_loader.sampler.get_state()
                    if valid_acc > self.best_acc:
                        self.best_acc = valid_acc
                        logger.info(f"New best valid acc found: {valid_acc}. Saving checkpoint...")
                        valid_result['best_acc'] = self.best_acc
                        torch.save(valid_result, os.path.join(self.config['save_dir'], 'checkpoint_best.pt'))
                    logger.info(f"------> Saving checkpoint...")
                    valid_result['best_acc'] = self.best_acc
                    torch.save(valid_result, os.path.join(self.config['save_dir'], 'checkpoint_last.pt'))
                    logger.info("------> Done.")
        return

    def _train_epoch(self, epoch):
        self.model.train()
        result = None
        loss_meter = AverageMeter()
        ground_truth = []
        predictions = []
        num_batches_per_print = len(self.train_loader) // self.config['num_prints']
        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(self.model, MLMBaseline):
                assert self.train_loader.dataset.model_type == 0, "Set dataset's model_type to 0 when using MLM."
                input_ids, token_type_ids, attention_mask, labels = batch['input_ids'], \
                                                                    batch['token_type_ids'], batch['attention_mask'], \
                                                                    batch['labels']
                input_ids, token_type_ids, attention_mask, labels = (input_ids.to(self.device),
                                                                     token_type_ids.to(self.device),
                                                                     attention_mask.to(self.device),
                                                                     labels.to(self.device)
                                                                     )
                outputs = self.model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
            else:
                raise RuntimeError(f"Baseline not implemented: {type(self.model)}")
            loss = self.criterion(outputs, labels)
            loss.backward()
            if self.config['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler:
                self.lr_scheduler.step()
            pred = torch.argmax(outputs, dim=-1)
            predictions.extend(pred.detach().cpu().numpy())
            ground_truth.extend(labels.detach().cpu().numpy())
            loss_meter.update(loss.item())
            result = {"loss": loss_meter.average(),
                      "lr": lr}
            if (batch_idx + 1) % num_batches_per_print == 0:
                logger.info(f"Training "
                            f"[{batch_idx * self.train_loader.batch_size}/{len(self.train_loader.dataset)} "
                            f"({(100. * batch_idx / len(self.train_loader)):.0f}%)] "
                            f"| Loss: {loss_meter.average():.5f} "
                            f"| Learning Rate: {lr}")
        acc = (ground_truth == predictions).sum()
        logger.info(f"Finished training epoch {epoch} "
                    f"| Loss: {loss_meter.average():.5f} "
                    f"| Accuracy: {acc:.4f} "
                    )
        result['acc'] = acc
        return result

    def _eval_epoch(self, epoch):
        self.model.eval()
        result = None
        loss_meter = AverageMeter()
        ground_truth = []
        predictions = []
        num_batches_per_print = len(self.valid_loader) // self.config['num_prints']
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                if isinstance(self.model, MLMBaseline):
                    assert self.valid_loader.dataset.model_type == 0, "Set dataset's model_type to 0 when using Bert."
                    input_ids, token_type_ids, attention_mask, labels = batch['input_ids'], \
                                batch['token_type_ids'], batch['attention_mask'], batch['labels']
                    input_ids, token_type_ids, attention_mask, labels = (input_ids.to(self.device),
                                                                         token_type_ids.to(self.device),
                                                                         attention_mask.to(self.device),
                                                                         labels.to(self.device)
                                                                         )
                    outputs = self.model(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                else:
                    raise RuntimeError("Baseline not implemented")
                loss = self.criterion(outputs, labels)
                pred = torch.argmax(outputs, dim=-1)
                predictions.extend(pred.detach().cpu().numpy())
                ground_truth.extend(labels.detach().cpu().numpy())
                loss_meter.update(loss.item())
                result = {"loss": loss_meter.average()
                          }
                if (batch_idx + 1) % num_batches_per_print == 0:
                    logger.info(f"Evaluating "
                                f"[{batch_idx * self.valid_loader.batch_size}/{len(self.valid_loader.dataset)} "
                                f"({(100. * batch_idx / len(self.valid_loader)):.0f}%)] "
                                f"| Loss: {loss_meter.average():.5f} "
                                )
        acc = (ground_truth == predictions).sum()
        logger.info(f"Finished training epoch {epoch} "
                    f"| Loss: {loss_meter.average():.5f} "
                    f"| Accuracy: {acc:.4f} "
                    )
        result['acc'] = acc
        return result

    def _load_model(self):
        checkpoint_path = os.path.join(self.config['save_dir'], 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.param_groups[0]['capturable'] = True
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_acc = checkpoint['best_acc']
        self.current_epoch = checkpoint['epoch']
        self.train_loader.sampler.set_state(checkpoint['train_sampler'])
        logger.info(f"------> Loaded checkpoint from {checkpoint_path}")
