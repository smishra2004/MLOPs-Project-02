import io
import sys
import tempfile

import torch
import torch.nn as nn
from torchvision import models

from src.configuration.aws_connection import S3Client
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging


class ChestXrayEstimator:
    """
    Mirrors: Proj1Estimator

    Key difference from Proj1Estimator:
      - Proj1Estimator uses aws_storage.load_model() which uses pickle.loads()
        → works for sklearn models
      - ChestXrayEstimator bypasses that method entirely
        → downloads raw .pth bytes from S3
        → loads with torch.load() which understands PyTorch format

    We never modify aws_storage.py — it still works for the other project.
    """

    def __init__(self, bucket_name: str, model_path: str) -> None:
        """
        Args:
            bucket_name (str): S3 bucket where the model is stored.
            model_path  (str): S3 key path e.g. 'cnn_model.pkl' or 'models/resnet50.pth'
        """
        self.bucket_name  = bucket_name
        self.model_path   = model_path
        self.s3           = SimpleStorageService()      # for is_model_present + upload
        self.s3_client    = S3Client().s3_client        # boto3 client for direct download
        self.loaded_model = None                        # lazy loaded on first predict call
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ──────────────────────────────────────────────────────────────────────────
    def is_model_present(self, model_path: str) -> bool:
        """
        Checks whether a model file exists at the given S3 key.
        Mirrors: Proj1Estimator.is_model_present()
        Uses aws_storage.s3_key_path_available() — this is safe, no pickle involved.
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
        Rebuilds the ResNet50 shell with our custom head.
        weights=None — we are about to pour in our own saved weights.
        Architecture must exactly match what was saved in ModelTrainer.
        """
        try:
            model       = models.resnet50(weights=None)
            in_features = model.fc.in_features          # 2048
            model.fc    = nn.Sequential(
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
        Downloads the .pth file from S3 using boto3 directly
        and loads it with torch.load().

        WHY we bypass aws_storage.load_model():
          aws_storage.load_model() does:
              model = pickle.loads(model_obj)   ← works for sklearn
          Our .pth file is NOT a pickle file — it is a PyTorch checkpoint.
          pickle.loads() crashes with UnpicklingError on it.

        FIX — download raw bytes with boto3, load with torch.load():
          s3_client.download_fileobj() → raw bytes into memory buffer
          torch.load(buffer)           → correctly reads PyTorch format

        Mirrors: Proj1Estimator.load_model() in intent,
                 but uses torch.load() instead of pickle.loads()
        """
        try:
            logging.info(
                f"Loading PyTorch model from S3: "
                f"{self.bucket_name}/{self.model_path}"
            )

            # ── Download raw .pth bytes directly into a memory buffer ─────
            # We use a temp file because torch.load needs seekable file-like
            # object, and S3 streaming body is not seekable
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                tmp_path = tmp.name

            # boto3 download_file → saves directly to temp file on disk
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=self.model_path,
                Filename=tmp_path,
            )
            logging.info("Model file downloaded from S3 to temp file.")

            # ── Build architecture shell + load weights ───────────────────
            model = self._build_model_architecture()
            model.load_state_dict(
                torch.load(tmp_path, map_location=self.device)
            )
            model = model.to(self.device)
            model.eval()

            # ── Clean up temp file ────────────────────────────────────────
            import os
            os.remove(tmp_path)
            logging.info("Model loaded from S3 successfully.")

            return model

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Uploads a local .pth file to S3.
        Mirrors: Proj1Estimator.save_model()
        Uses aws_storage.upload_file() — safe, no pickle involved.

        Args:
            from_file (str): Local path to the saved model weights.
            remove    (bool): If True, deletes local file after upload.
        """
        try:
            logging.info(
                f"Uploading model to S3: "
                f"{self.bucket_name}/{self.model_path}"
            )
            self.s3.upload_file(
                from_filename=from_file,
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

        Lazy loads the model from S3 on first call, reuses it after.

        Args:
            image_tensor: shape [N, 3, 224, 224] — resized and normalised

        Returns:
            torch.Tensor: predicted class indices [N] — 0=NORMAL, 1=PNEUMONIA
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                outputs      = self.loaded_model(image_tensor)
                _, preds     = torch.max(outputs, dim=1)

            return preds

        except Exception as e:
            raise MyException(e, sys)