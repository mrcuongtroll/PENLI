from definitions import *
import argparse
from utils.config import ConfigParser
import model.model as model_module
import data.data_loader as data_loader
from trainer.trainer import Trainer
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
    if args.finetune_critic:
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoint_best.pt')
        assert os.path.exists(checkpoint_path), "Trained actor checkpoint does not exist. Cannot finetune critic."
        logger.info(f"-----> Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        logger.info(f"-----> Done.")
    train_loader = config.init_obj(data_loader, 'train_loader', tokenizer=model.tokenizer, use_explanation=False)
    valid_loader = config.init_obj(data_loader, 'valid_loader', tokenizer=model.tokenizer, use_explanation=False)
    # test_loader = config.init_obj(data_loader, 'test_loader', tokenizer=model.tokenizer, use_explanation=False)
    optimizer = config.init_obj(optim, 'optimizer', model.parameters())
    criterion = config.init_obj(nn, "loss")
    trainer = Trainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      config=config,
                      finetune_critic=args.finetune_critic,
                      device=device)
    trainer.train(resume=args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training?")
    parser.add_argument("--config", default=DEFAULT_TRAINING_CONFIG, type=str, help="Path to config file "
                                                                                    "(json format).")
    parser.add_argument('--device',
                        default=None,
                        nargs='?',
                        choices=['cuda', 'cpu'],
                        help='Device to train the model on (cuda/cpu)')
    parser.add_argument('--finetune_critic',
                        default=False,
                        action='store_true',
                        help='If True, use the current best checkpoint to finetune critic.')
    args = parser.parse_args()
    main(args)
