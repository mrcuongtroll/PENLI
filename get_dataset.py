import os
from data.download import download_and_extract
from definitions import *


if __name__ == '__main__':
    if not os.path.exists(SNLI_DIR):
        print("Getting SNLI dataset...")
        download_and_extract(url=SNLI_URL, destination_dir=DATASETS_DIR)
    else:
        print(f"SNLI dataset already exists and can be found in {SNLI_DIR}")
