from definitions import *
import argparse
from utils.config import ConfigParser
import model.rl.actor_critic as rl_model_module
import model.model as model_module
import data.data_loader as data_loader
from trainer.rl_trainer import RLTrainer
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
    torch.manual_seed(args.seed)
    config = ConfigParser(args.config)
    config.init_obj(logger_module, "logging")
    if not args.device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model = config.init_obj(model_module, "model")
    if args.best_ckpt:
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoint_best.pt')
    else:
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoint_last.pt')
    if not os.path.exists(checkpoint_path):
        logger.critical(f"The checkpoint {checkpoint_path} doesn't exist for the config {args.config}. "
                        f"Train a model with the given config using train.py")
        exit()
    else:
        logger.info(f"-----> Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        logger.info(f"-----> Done.")
    rl_module = config.init_obj(rl_model_module, "rl", model=model, device=device)
    train_loader = config.init_obj(data_loader, 'train_loader', tokenizer=rl_module.model.tokenizer,
                                   use_explanation=True, seed=args.seed)
    valid_loader = config.init_obj(data_loader, 'valid_loader', tokenizer=rl_module.model.tokenizer,
                                   use_explanation=True, seed=args.seed)
    test_loader = config.init_obj(data_loader, 'test_loader', tokenizer=rl_module.model.tokenizer,
                                  use_explanation=True, seed=args.seed)
    optimizer = config.init_obj(optim, 'optimizer', rl_module.parameters())
    trainer = RLTrainer(module=rl_module,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        config=config,
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
    parser.add_argument('--best_ckpt', default=False, action='store_true', help='use the best checkpoint or the latest')
    parser.add_argument("--seed", default=69420, type=int, help='random seed')
    args = parser.parse_args()
    main(args)
