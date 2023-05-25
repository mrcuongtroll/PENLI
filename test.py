from definitions import *
import argparse
from utils.config import ConfigParser
from utils.utils import evaluate_model
import model.model as model_module
import data.data_loader as data_loader
import logger.logger as logger_module
import torch
import torch.nn as nn
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
    test_loader = config.init_obj(data_loader, 'test_loader', tokenizer=model.tokenizer, use_explanation=True)
    criterion = config.init_obj(nn, "loss")
    results = evaluate_model(model=model,
                             data_loader=test_loader,
                             criterion=criterion,
                             config=config,
                             use_explanation=args.use_explanation,
                             device=device)
    logger.info(f"*********************FINISHED EVALUATING MODEL**********************")
    logger.info(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_TRAINING_CONFIG, type=str, help="Path to config file "
                                                                                    "(json format).")
    parser.add_argument('--device',
                        default=None,
                        nargs='?',
                        choices=['cuda', 'cpu'],
                        help='Device to train the model on (cuda/cpu)')
    parser.add_argument('--use_explanation', default=False, action='store_true',
                        help='to concatenate the explanation to the input or not')
    parser.add_argument('--best_ckpt', default=False, action='store_true', help='use the best checkpoint or the latest')
    args = parser.parse_args()
    main(args)
