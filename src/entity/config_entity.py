from dataclasses import dataclass
import os
from src.constants import *   # e.g. "artifacts"
from datetime import datetime
from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import os
from src.constants import ARTIFACT_DIR, S3_BUCKET_NAME, S3_MODEL_KEY_PATH
 
 

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
 
@dataclass
class DataTransformationConfig:
    """What the data transformation component NEEDS (inputs / settings)."""
 
    batch_size:  int = 32   # 32 is standard; reduce to 16 if GPU runs out of memory
    num_workers: int = 0    # parallel data loading; set to 0 if on Windows
 
from dataclasses import dataclass
import os
from src.constants import ARTIFACT_DIR
 
 
@dataclass
class ModelTrainerConfig:
    """What the model trainer component NEEDS."""
 
    num_epochs:      int   = 1      # increase if you move to GPU later
    learning_rate:   float = 1e-4   # Adam LR — 1e-4 is standard for fine-tuning
    patience:        int   = 3      # early stopping: stop after 3 non-improving epochs
 
    model_save_path: str   = os.path.join(
        ARTIFACT_DIR, 'model_trainer', 'resnet50_best.pth'
    )
 
@dataclass
class ModelEvaluationConfig:
    """What the model evaluation component NEEDS."""
 
    # S3 bucket where the production model lives
    bucket_name       : str = S3_BUCKET_NAME
 
    # S3 key path of the production model
    # e.g. "models/resnet50_best.pth"
    s3_model_key_path : str = S3_MODEL_KEY_PATH
 
    # How much AUC improvement is needed to accept new model
    # 0.0 means any improvement is accepted (mirrors reference project)
    changed_threshold : float = 0.0
 
 
@dataclass
class ModelPusherConfig:
    """What the model pusher component NEEDS."""
 
    # Same bucket and key as evaluator — pusher overwrites production model
    bucket_name       : str = S3_BUCKET_NAME
    s3_model_key_path : str = S3_MODEL_KEY_PATH