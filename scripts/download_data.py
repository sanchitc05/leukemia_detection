"""
Prepares the extracted ALL_IDB1 dataset for the Leukemia Detection project.
Moves the extracted files into the expected raw data directory.
"""

import shutil
import logging
from pathlib import Path
from src.utils import setup_logging
from src.model_config import config

# Setup logging
setup_logging(log_file="results/logs/download.log")
logger = logging.getLogger(__name__)


def prepare_dataset(source_dir: str, destination_dir: str):
    """
    Move dataset from extracted location to project raw directory.

    Args:
        source_dir: Path where dataset was extracted
        destination_dir: Target path inside the project
    """
    source = Path(source_dir)
    destination = Path(destination_dir)

    if not source.exists():
        logger.error(f"Source directory does not exist: {source}")
        return

    destination.mkdir(parents=True, exist_ok=True)

    # Move or copy files
    for item in source.iterdir():
        dest_item = destination / item.name
        if item.is_file():
            shutil.copy2(item, dest_item)
        elif item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)

    logger.info(f"Dataset prepared at: {destination}")


def main():
    # Define paths
    destination_path = config.data_config['dataset_path']
    
    # Prompt for extracted dataset path
    extracted_path = input("Enter the full path to your extracted ALL_IDB1 dataset: ").strip()

    logger.info("Preparing dataset...")
    prepare_dataset(source_dir=extracted_path, destination_dir=destination_path)
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()