import logging
import json
import os
from definitions import *


logger = logging.getLogger(name=__name__)


# Classes
class ConfigParser:

    def __init__(self, file_path: str = None):
        """
        Config can be loaded from file_path to create a ConfigParser object.
        :param file_path: Path to the config file.
        """
        self.config = {}
        if file_path:
            self.load(file_path)

    def init_obj(self, module, obj, *args, **kwargs):
        """
        Initialize object obj from module, given the config of the obj and option args, kwargs.
        :param module: The module whose object to be initialized.
        :param obj: The object name in the config file.
        :param args: Optional arguments.
        :param kwargs: Optional keyword arguments.
        :return: Initialized object.
        """
        class_name = self.config[obj]['type']
        module_kwargs = self.config[obj]['kwargs']
        module_kwargs.update(kwargs)
        return getattr(module, class_name)(*args, **module_kwargs)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def load(self, file_path: str):
        """
        Load config from a config file.
        :param file_path: Path to the json file containing the config.
        :return:
        """
        logger.info(f"------> Loading config from {file_path}...")
        with open(file_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"------> Done.")
        save_dir = os.path.abspath(self.config["save_dir"])
        self.config["save_dir"] = os.path.join(save_dir, self.config['name'])
        if "logging" in self.config.keys():
            self.config["logging"]['kwargs']['save_dir'] = self.config['save_dir']
        os.makedirs(self.config['save_dir'], exist_ok=True)

    def write(self, file_path: str = None):
        """
        Save the current config to a file.
        :param file_path: The file path to save the config into. Must be provided if self.config['save_dir'] is not
                          available.
        :return:
        """
        if file_path is None:
            if 'save_dir' not in self.config.keys() or self.config['save_dir'] is None:
                raise RuntimeError("The directory to save the config does not exist and is not provided.")

            else:
                os.makedirs(self.config['save_dir'], exist_ok=True)
                with open(os.path.join(self.config['save_dir'], 'config.json'), 'w') as f:
                    json.dump(self.config, f)
        else:
            if os.path.isfile(file_path):
                directory, file_name = os.path.split(file_path)
                self.config['save_dir'] = directory
                os.makedirs(directory, exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(self.config, f)
            else:
                self.config['save_dir'] = file_path
                os.makedirs(file_path, exist_ok=True)
                with open(os.path.join(file_path, 'config.json'), 'w') as f:
                    json.dump(self.config, f)
