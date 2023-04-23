import wget
from zipfile import ZipFile
import logging
from definitions import *


logger = logging.getLogger(name=__name__)


# Functions
def download_and_extract(url: str = MINION_URL, destination_dir: str = DATASETS_DIR):
    """
    This function downloads a zip file from a given url into a destination directory then unzip it.
    :param url: (type: str) The url to download the zip file from.
    :param destination_dir: (type: str) The directory to store the downloaded zip file and its extracted content.
    :return: None
    """
    logger.info(f'Downloading data from "{url}" to "{destination_dir}"...')
    filename = wget.download(url=url, out=destination_dir)
    logger.info("---> Done.")
    logger.info(f"Extracting downloaded file...")
    with ZipFile(filename, 'r') as z:
        z.extractall(path=destination_dir)
    logger.info("---> Done.")
    return
