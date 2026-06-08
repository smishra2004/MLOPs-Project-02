import os
import sys

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.aws_data_access import ChestXrayData

print("FILE STARTED")


class DataIngestion:
    """
    Orchestrates the full data ingestion pipeline for the chest X-ray
    deep learning project:
      1. Downloads the zipped dataset from S3.
      2. Extracts it locally, preserving train/test/val structure.
      3. Returns a DataIngestionArtifact with paths to each split.

    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def download_data_from_s3(self) -> str:
        """
        Downloads the dataset zip from S3 to the local zip file path
        defined in DataIngestionConfig.

        Returns:
            str: Local path of the downloaded zip file.
        """
        try:
            logging.info("Initiating S3 dataset download.")
            chest_xray_data = ChestXrayData()

            local_zip_path = chest_xray_data.download_zip_from_s3(
                local_zip_path=self.data_ingestion_config.local_zip_file_path
            )

            logging.info(f"Dataset zip available at: {local_zip_path}")
            return local_zip_path

        except Exception as e:
            raise MyException(e, sys)

    def extract_data(self, local_zip_path: str) -> str:
        """
        Extracts the downloaded zip file to the raw data directory
        defined in DataIngestionConfig.

        Args:
            local_zip_path (str): Path to the downloaded zip file.

        Returns:
            str: Root directory of the extracted dataset.
        """
        try:
            logging.info("Initiating dataset extraction.")
            chest_xray_data = ChestXrayData()

            extract_dir = chest_xray_data.extract_zip(
                local_zip_path=local_zip_path,
                extract_dir=self.data_ingestion_config.raw_data_dir,
            )

            logging.info(f"Dataset extracted to: {extract_dir}")
            return extract_dir

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Entry point for the data ingestion stage. Runs download → extract,
        then packages the resulting split paths into a DataIngestionArtifact.

        Returns:
            DataIngestionArtifact: Artifact holding paths to train/test/val splits.
        """
        try:
            logging.info(
                "========== Entered initiate_data_ingestion — DataIngestion ==========")

            # Step 1: Download zip from S3
            local_zip_path = self.download_data_from_s3()
            logging.info("S3 download complete.")

            # Step 2: Extract zip preserving train/test/val/NORMAL/PNEUMONIA layout
            extract_dir = self.extract_data(local_zip_path)
            logging.info("Extraction complete.")

            # Step 3: Build split paths from the extracted root directory
            # train_dir = os.path.join(extract_dir, "train")
            # test_dir  = os.path.join(extract_dir, "test")
            # val_dir   = os.path.join(extract_dir, "val")
            
            # Dataset root after extraction
            dataset_root = os.path.join(extract_dir, "chest_xray")

            # Build split paths
            train_dir = os.path.join(dataset_root, "train")
            test_dir = os.path.join(dataset_root, "test")
            val_dir = os.path.join(dataset_root, "val")

            logging.info(f"Train Dir: {train_dir}")
            logging.info(f"Test Dir: {test_dir}")
            logging.info(f"Val Dir: {val_dir}")

            # Step 4: Package into artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_dir=train_dir,
                test_dir=test_dir,
                val_dir=val_dir,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info(
                "========== Exited initiate_data_ingestion — DataIngestion ==========")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)
