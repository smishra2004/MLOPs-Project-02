import sys
import torch
import torch.nn as nn
from torchvision import models

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging


class ChestXrayEstimator:
    """
    Mirrors: Proj1Estimator

    Responsible for:
      - Checking if a model exists in S3
      - Loading a ResNet50 model from S3
      - Saving a local model to S3
      - Running inference on a batch of image tensors

    The key difference from Proj1Estimator:
      - Proj1Estimator loads a sklearn model and calls .predict(dataframe)
      - ChestXrayEstimator loads a PyTorch ResNet50 and calls .predict(tensor)
    """

    def __init__(self, bucket_name: str, model_path: str) -> None:
        """
        Args:
            bucket_name (str): S3 bucket where the model is stored.
            model_path  (str): S3 key path of the model file e.g. 'models/resnet50_best.pth'
        """
        self.bucket_name   = bucket_name
        self.model_path    = model_path
        self.s3            = SimpleStorageService()
        self.loaded_model  = None   # lazy loaded on first predict call

    # ──────────────────────────────────────────────────────────────────────────
    def is_model_present(self, model_path: str) -> bool:
        """
        Checks whether a model file exists at the given S3 key.
        Mirrors: Proj1Estimator.is_model_present()

        Returns:
            bool: True if model exists in S3, False otherwise.
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path,
            )
        except MyException as e:
            logging.warning(f"Could not check model presence in S3: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    def _build_model_architecture(self) -> nn.Module:
        """
        Rebuilds the ResNet50 architecture with the same custom head
        used during training. Required before loading saved weights.

        Architecture must match exactly what was saved in ModelTrainer:
            Linear(2048 → 256) → ReLU → Dropout(0.4) → Linear(256 → 2)
        """
        try:
            model = models.resnet50(weights=None)   # no pretrained weights, we load ours
            in_features = model.fc.in_features       # 2048
            model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 2)
            )
            return model
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def load_model(self) -> nn.Module:
        """
        Downloads the model weights from S3 and loads them into
        the ResNet50 architecture.
        Mirrors: Proj1Estimator.load_model()

        Returns:
            nn.Module: ResNet50 in eval mode with weights loaded from S3.
        """
        try:
            logging.info(f"Loading model from S3: {self.bucket_name}/{self.model_path}")

            # Download .pth file from S3 into memory
            model_obj = self.s3.load_model(
                model_name=self.model_path,
                bucket_name=self.bucket_name,
            )

            # Rebuild architecture and load state dict
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model  = self._build_model_architecture()
            model.load_state_dict(
                torch.load(model_obj, map_location=device)
            )
            model = model.to(device)
            model.eval()

            logging.info("Model loaded from S3 successfully.")
            return model

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Uploads a local .pth file to S3.
        Mirrors: Proj1Estimator.save_model()

        Args:
            from_file (str): Local path to the saved model weights.
            remove    (bool): If True, deletes local file after upload.
                              Default False — keep local copy.
        """
        try:
            logging.info(
                f"Uploading model to S3: {self.bucket_name}/{self.model_path}"
            )
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
            logging.info("Model uploaded to S3 successfully.")
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs inference on a batch of preprocessed image tensors.
        Mirrors: Proj1Estimator.predict(dataframe)

        Lazy loads the model from S3 on first call, then reuses it.

        Args:
            image_tensor (torch.Tensor): Batch of images shape [N, 3, 224, 224]
                                         already resized and normalised.

        Returns:
            torch.Tensor: Predicted class indices [N] — 0=NORMAL, 1=PNEUMONIA
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                outputs = self.loaded_model(image_tensor)
                _, preds = torch.max(outputs, dim=1)

            return preds

        except Exception as e:
            raise MyException(e, sys)