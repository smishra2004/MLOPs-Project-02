import os
import sys
import zipfile

from src.configuration.aws_connection import S3Client
from src.constants import S3_BUCKET_NAME, S3_DATA_KEY
from src.exception import MyException
from src.logger import logging


class ChestXrayData:
    """
    Handles downloading and extracting the chest X-ray dataset
    from an AWS S3 bucket. 
    """

    def __init__(self) -> None:
        try:
            self.s3_client = S3Client()
        except Exception as e:
            raise MyException(e, sys)

    def download_zip_from_s3(self, local_zip_path: str) -> str:
        """
        Downloads the zipped dataset from the configured S3 bucket
        to a local path.

        Args:
            local_zip_path (str): Local file path where the zip will be saved.

        Returns:
            str: The local path of the downloaded zip file.
        """
        try:
            logging.info(
                f"Starting download from S3 bucket: '{S3_BUCKET_NAME}', "
                f"key: '{S3_DATA_KEY}'"
            )

            os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)

            self.s3_client.s3_client.download_file(
                Bucket=S3_BUCKET_NAME,
                Key=S3_DATA_KEY,
                Filename=local_zip_path,
            )

            logging.info(f"Dataset zip downloaded successfully to: {local_zip_path}")
            return local_zip_path

        except Exception as e:
            raise MyException(e, sys)

    def extract_zip(self, local_zip_path: str, extract_dir: str) -> str:
        """
        Extracts the downloaded zip file to the specified directory,
        preserving the train/test/val → normal/pneumonia folder structure.

        Args:
            local_zip_path (str): Path to the local zip file.
            extract_dir (str)   : Directory where contents will be extracted.

        Returns:
            str: The extraction directory path.
        """
        try:
            logging.info(f"Extracting zip file: {local_zip_path} -> {extract_dir}")

            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            dataset_root = os.path.join(extract_dir, "chest_xray")

            logging.info(
                f"Extraction complete. "
                f"Expected structure: {dataset_root}/[train|test|val]/[NORMAL|PNEUMONIA]/"
            )

            self._validate_extracted_structure(dataset_root)

            return extract_dir

        except Exception as e:
            raise MyException(e, sys)

    def _validate_extracted_structure(self, extract_dir: str) -> None:
        """
        Validates that the expected folder structure exists after extraction.
        Logs a warning if any expected split or class folder is missing.

        Args:
            extract_dir (str): Root directory of the extracted dataset.
        """
        try:
            expected_splits = ["train", "test", "val"]
            expected_classes = ["NORMAL", "PNEUMONIA"]

            for split in expected_splits:
                split_path = os.path.join(extract_dir, split)
                if not os.path.isdir(split_path):
                    logging.warning(f"Expected split folder not found: {split_path}")
                    continue

                for cls in expected_classes:
                    cls_path = os.path.join(split_path, cls)
                    if not os.path.isdir(cls_path):
                        logging.warning(f"Expected class folder not found: {cls_path}")
                    else:
                        n_images = len(os.listdir(cls_path))
                        logging.info(
                            f"[{split}/{cls}] — {n_images} images found."
                        )

        except Exception as e:
            raise MyException(e, sys)