import logging
import logging.config
import json
import os
from definitions import *


# Functions
def setup_logging(save_dir: str = LOGS_DIR, log_config: str = DEFAULT_LOGGING_CONFIG):
    """
    Setup logging configuration. The config file must be in json format. If a config file is not given, or the file is
    not json, the basic config with NOTSET level will be used instead.
    :param log_config: (type: str) Path to the config file.
    :param save_dir: (type: str) Path to the directory that stores the logs.
    :return: None
    """
    if log_config is None or '.json' not in log_config:
        logging.basicConfig(level=logging.NOTSET)
    else:
        os.makedirs(save_dir, exist_ok=True)
        with open(log_config, 'r') as f:
            config = json.load(f)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = os.path.join(save_dir, handler['filename'])
        logging.config.dictConfig(config)
    return
