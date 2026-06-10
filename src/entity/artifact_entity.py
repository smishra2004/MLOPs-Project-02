from dataclasses import dataclass
import os
from src.constants import *   # e.g. "artifacts"
from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any
 

@dataclass
class DataIngestionArtifact:
    """
    What the data ingestion component PRODUCES (outputs for next stage).
    """
    train_dir: str   # .../raw/chest_xray/train/
    test_dir:  str   # .../raw/chest_xray/test/
    val_dir:   str   # .../raw/chest_xray/val/
    
@dataclass
class DataTransformationArtifact:
    """What the data transformation component PRODUCES (outputs for model trainer)."""
 
    train_loader:  DataLoader     # augmented + weighted sampled
    test_loader:   DataLoader     # clean, no augmentation
    val_loader:    DataLoader     # clean, no augmentation
    class_weights: torch.Tensor   # [weight_NORMAL, weight_PNEUMONIA] for loss function
 
 
@dataclass
class ModelTrainerArtifact:
    """What the model trainer component PRODUCES (feeds into ModelEvaluator)."""
 
    model_save_path: str    # path to saved resnet50_best.pth
    test_accuracy:   float  # accuracy on test set after training
    test_loss:       float  # loss on test set after training
    history:         Any    # dict → train_loss, train_acc, val_loss, val_acc per epoch
    

@dataclass
class ModelEvaluationArtifact:
    """What the model evaluation component PRODUCES (feeds into ModelPusher)."""
 
    is_model_accepted  : bool   # True if trained model beats production AUC
    s3_model_path      : str    # S3 key of the current production model
    trained_model_path : str    # local path of the newly trained model
    changed_accuracy   : float  # AUC difference: trained_auc - production_auc
 
 
@dataclass
class ModelPusherArtifact:
    """What the model pusher component PRODUCES."""
 
    bucket_name   : str    # S3 bucket where model was pushed
    s3_model_path : str    # S3 key path of the pushed model
    is_pushed     : bool   # True if model was actually uploaded