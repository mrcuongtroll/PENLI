import os
from data.download import download_and_extract
from definitions import *


if __name__ == '__main__':
    if not os.path.exists(MINION_DIR):
        print("Getting MINION dataset...")
        download_and_extract(url=MINION_URL, destination_dir=DATASETS_DIR)
    else:
        print(f"MINION dataset already exists and can be found in {MINION_DIR}")
