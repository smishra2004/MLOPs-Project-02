from dataclasses import dataclass
import os
from src.constants import *   # e.g. "artifacts"
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = PIPELINE_NAME
    artifact_dir:str = os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()
 
# @dataclass
# class DataIngestionConfig:
#     """
#     What the data ingestion component NEEDS (inputs / settings).
#     """
#     # Where the downloaded zip will be saved locally
#     local_zip_file_path: str = os.path.join(
#         ARTIFACT_DIR, "data_ingestion", "raw", "chest_xray.zip"
#     )
 
#     # Root directory where the zip is extracted
#     # Final on-disk layout after extraction:
#     #   raw_data_dir/
#     #       train/NORMAL/*.jpeg
#     #       train/PNEUMONIA/*.jpeg
#     #       test/NORMAL/*.jpeg
#     #       test/PNEUMONIA/*.jpeg
#     #       val/NORMAL/*.jpeg
#     #       val/PNEUMONIA/*.jpeg
#     raw_data_dir: str = os.path.join(
#         ARTIFACT_DIR, "data_ingestion", "raw", "chest_xray"
#     )

@dataclass
class DataIngestionConfig:

    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        "data_ingestion"
    )

    local_zip_file_path: str = os.path.join(
        data_ingestion_dir,
        "raw",
        "chest_xray.zip"
    )

    raw_data_dir: str = os.path.join(
        data_ingestion_dir,
        "raw"
    )