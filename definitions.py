"""
This file contains constants frequently used by other modules.
"""
import os


# Logging
LOGS_DIR = os.path.abspath('./logs')
DEFAULT_LOGGING_CONFIG = os.path.abspath('./logger/logging_config.json')


# Data
DATASETS_DIR = os.path.abspath('./datasets')
MINION_URL = 'https://drive.google.com/u/0/uc?id=11eVivQxQ6zeRNXIQ7jA-xq8zfYCYDqkJ&export=download'
MINION_DIR = os.path.join(DATASETS_DIR, 'MINION')
MINION_LANGUAGES = ('english', 'hindi', 'japanese', 'korean', 'polish', 'portuguese', 'spanish', 'turkisk')
MINION_LANGUAGE_CODES = ('en', 'hi', 'ja', 'ko', 'pl', 'pt', 'es', 'tr')
MINION_LABEL_TO_ID = {
               'O': 0,
               'B_Life:Be-Born': 1, 'I_Life:Be-Born': 2,
               'B_Life:Marry': 3, 'I_Life:Marry': 4,
               'B_Life:Divorce': 5, 'I_Life:Divorce': 6,
               'B_Life:Injure': 7, 'I_Life:Injure': 8,
               'B_Life:Die': 9, 'I_Life:Die': 10,
               'B_Movement:Transport': 11, 'I_Movement:Transport': 12,
               'B_Transaction:Transfer-Ownership': 13, 'I_Transaction:Transfer-Ownership': 14,
               'B_Transaction:Transfer-Money': 15, 'I_Transaction:Transfer-Money': 16,
               'B_Conflict:Attack': 17, 'I_Conflict:Attack': 18,
               'B_Conflict:Demonstrate': 19, 'I_Conflict:Demonstrate': 20,
               'B_Contact:Meet': 21, 'I_Contact:Meet': 22,
               'B_Contact:Phone-Write': 23, 'I_Contact:Phone-Write': 24,
               'B_Personnel:Start-Position': 25, 'I_Personnel:Start-Position': 26,
               'B_Personnel:End-Position': 27, 'I_Personnel:End-Position': 28,
               'B_Justice:Arrest-Jail': 29, 'I_Justice:Arrest-Jail': 30,
               'B_Business:START-ORG': 31, 'I_Business:START-ORG': 32
}
MINION_ID_TO_LABEL = {idx: label for (label, idx) in MINION_LABEL_TO_ID.items()}
MINION_ID_TO_LABEL[-100] = "[PAD]"
DATASET_TYPES = ('train', 'training', 'test', 'testing', 'dev', 'valid', 'development', 'validation')


# Models
TRAINING_CONFIG_DIR = os.path.abspath('./configs')
DEFAULT_TRAINING_CONFIG = os.path.join(TRAINING_CONFIG_DIR, 'default.json')


# Initialization
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TRAINING_CONFIG_DIR, exist_ok=True)
