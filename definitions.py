"""
This file contains constants frequently used by other modules.
"""
import os


# Logging
LOGS_DIR = os.path.abspath('./logs')
DEFAULT_LOGGING_CONFIG = os.path.abspath('./logger/logging_config.json')


# Data
DATASETS_DIR = os.path.abspath('./datasets')
SNLI_DIR = os.path.join(DATASETS_DIR, 'snli_1.0')
SNLI_URL = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
E_SNLI_DIR = os.path.join(DATASETS_DIR, 'e-SNLI')
DATASET_TYPES = ('train', 'training', 'test', 'testing', 'dev', 'valid', 'development', 'validation')


# Models
TRAINING_CONFIG_DIR = os.path.abspath('./configs')
DEFAULT_TRAINING_CONFIG = os.path.join(TRAINING_CONFIG_DIR, 'default.json')


# Initialization
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TRAINING_CONFIG_DIR, exist_ok=True)
