from dataclasses import dataclass
import os
from src.constants import *   # e.g. "artifacts"

@dataclass
class DataIngestionArtifact:
    """
    What the data ingestion component PRODUCES (outputs for next stage).
    """
    train_dir: str   # .../raw/chest_xray/train/
    test_dir:  str   # .../raw/chest_xray/test/
    val_dir:   str   # .../raw/chest_xray/val/
 