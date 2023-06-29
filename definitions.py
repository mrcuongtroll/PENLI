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
MLM_LABEL_MAPPING = {'neutral': 'Maybe', 'contradiction': 'No', 'entailment': 'Yes'}
GENERATIVE_LABEL_MAPPING = {'neutral': 'neutral', 'contradiction': 'false', 'entailment': 'true'}
IGNORE_ID = -100


# Models
TRAINING_CONFIG_DIR = os.path.abspath('./configs')
DEFAULT_TRAINING_CONFIG = os.path.join(TRAINING_CONFIG_DIR, 'default_ed.json')
PLM = {0: 'bert-base-uncased',     # 110M
       1: 'gpt2',                  # 124M
       2: 'google/flan-t5-base'    # 250M
       }
"""
Flan-T5 special tokens: eos: </s>,
                        unk: <unk>,
                        pad: <pad>,
                        
"""


# Initialization
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TRAINING_CONFIG_DIR, exist_ok=True)
