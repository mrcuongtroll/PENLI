from definitions import *
import argparse
from utils.config import ConfigParser
import model.model as model_module
import data.data_loader as data_loader
from trainer.baseline_trainer import BaselineTrainer
import logger.logger as logger_module
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import logging


logger = logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.exception("Uncaught exception encountered", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def main(args):
    config = ConfigParser(args.config)
    config.init_obj(logger_module, "logging")
    if not args.device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model = config.init_obj(model_module, "model")
    train_loader = config.init_obj(data_loader, 'train_loader', tokenizer=model.tokenizer)
    valid_loader = config.init_obj(data_loader, 'valid_loader', tokenizer=model.tokenizer)
    optimizer = config.init_obj(optim, 'optimizer', model.parameters())
    criterion = config.init_obj(nn, "loss")
    trainer = BaselineTrainer(model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              train_loader=train_loader,
                              valid_loader=valid_loader,
                              config=config,
                              device=device)
    trainer.train(resume=args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training?")
    parser.add_argument("--config", type=str, help="Path to config file (json format).")
    parser.add_argument('--device',
                        default=None,
                        nargs='?',
                        choices=['cuda', 'cpu'],
                        help='Device to train the model on (cuda/cpu)')
    args = parser.parse_args()
    main(args)
