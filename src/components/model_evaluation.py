import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import roc_auc_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.entity.chest_xray_s3_estimator import ChestXrayEstimator
from src.exception import MyException
from src.logger import logging


@dataclass
class EvaluateModelResponse:
    """
    Mirrors: EvaluateModelResponse from reference project.
    Holds comparison results between newly trained model
    and the current production model in S3.

    AUC used instead of F1 because:
      - AUC is threshold-independent (better for imbalanced medical data)
      - Captures false negative rate which is critical for pneumonia detection
    """
    trained_model_auc  : float
    best_model_auc     : float          # 0.0 if no production model exists yet
    is_model_accepted  : bool           # True if new model beats production
    difference         : float          # trained_auc - best_auc


class ModelEvaluation:
    """
    Mirrors: ModelEvaluation from reference project.

    Compares the newly trained ResNet50 against the current
    production model stored in S3.

    If no production model exists yet → new model is always accepted.
    If production model exists → new model must have higher AUC to be accepted.
    """

    def __init__(
        self,
        model_eval_config           : ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact      : ModelTrainerArtifact,
    ) -> None:
        try:
            self.model_eval_config            = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact       = model_trainer_artifact
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def get_best_model(self) -> Optional[ChestXrayEstimator]:
        """
        Fetches the current production model from S3 if it exists.
        Mirrors: ModelEvaluation.get_best_model()

        Returns:
            ChestXrayEstimator if a production model is found, else None.
        """
        try:
            estimator = ChestXrayEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path,
            )
            if estimator.is_model_present(
                model_path=self.model_eval_config.s3_model_key_path
            ):
                logging.info("Production model found in S3.")
                return estimator

            logging.info("No production model found in S3 — first deployment.")
            return None

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_auc(
        self,
        model    : nn.Module,
        loader   : torch.utils.data.DataLoader,
    ) -> float:
        """
        Computes AUC score for a given model on a given DataLoader.

        Why AUC over accuracy?
          Accuracy can be misleading with imbalanced data (2.89x ratio).
          AUC = 1.0 means perfect separation, 0.5 means random guessing.
          It also captures the false negative rate — missing pneumonia
          is the worst clinical mistake.

        Args:
            model  : PyTorch model in eval mode.
            loader : DataLoader (test set).

        Returns:
            float: AUC score between 0 and 1.
        """
        try:
            model.eval()
            all_probs  = []
            all_labels = []

            with torch.no_grad():
                for images, labels in loader:
                    images  = images.to(self.device)
                    outputs = model(images)
                    probs   = torch.softmax(outputs, dim=1)[:, 1]  # P(PNEUMONIA)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.numpy())

            auc = roc_auc_score(all_labels, all_probs)
            return float(auc)

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Core evaluation logic.
        Mirrors: ModelEvaluation.evaluate_model()

        Steps:
          1. Load newly trained model from local artifact path
          2. Compute its AUC on the test set
          3. Check if a production model exists in S3
          4. If yes → compute production model AUC on same test set
          5. Compare and decide is_model_accepted

        Returns:
            EvaluateModelResponse with AUC scores and acceptance decision.
        """
        try:
            logging.info("Starting model evaluation.")

            # ── Step 1: Load newly trained model ──────────────────────────
            logging.info(
                f"Loading trained model from: "
                f"{self.model_trainer_artifact.model_save_path}"
            )
            from torchvision import models

            trained_model = models.resnet50(weights=None)
            in_features   = trained_model.fc.in_features
            trained_model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 2)
            )
            trained_model.load_state_dict(
                torch.load(
                    self.model_trainer_artifact.model_save_path,
                    map_location=self.device,
                )
            )
            trained_model = trained_model.to(self.device)
            trained_model.eval()
            logging.info("Trained model loaded.")

            # ── Step 2: Compute trained model AUC ─────────────────────────
            trained_model_auc = self._compute_auc(
                model=trained_model,
                loader=self.data_transformation_artifact.test_loader,
            )
            logging.info(f"Trained model AUC: {trained_model_auc:.4f}")

            # ── Step 3: Check for production model in S3 ──────────────────
            best_model_auc = 0.0
            best_model     = self.get_best_model()

            if best_model is not None:
                # ── Step 4: Compute production model AUC ──────────────────
                logging.info("Computing AUC for production model from S3.")
                production_model = best_model.load_model()
                best_model_auc   = self._compute_auc(
                    model=production_model,
                    loader=self.data_transformation_artifact.test_loader,
                )
                logging.info(
                    f"Production model AUC: {best_model_auc:.4f}  |  "
                    f"Trained model AUC: {trained_model_auc:.4f}"
                )
            else:
                logging.info(
                    "No production model in S3 — "
                    "trained model accepted by default."
                )

            # ── Step 5: Acceptance decision ───────────────────────────────
            is_model_accepted = trained_model_auc > best_model_auc
            difference        = trained_model_auc - best_model_auc

            result = EvaluateModelResponse(
                trained_model_auc=trained_model_auc,
                best_model_auc=best_model_auc,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )
            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Entry point for the model evaluation component.
        Mirrors: ModelEvaluation.initiate_model_evaluation()

        Returns:
            ModelEvaluationArtifact with acceptance decision and paths.
        """
        try:
            logging.info(
                "========== Entered initiate_model_evaluation — ModelEvaluation =========="
            )

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.model_save_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(
                f"Model evaluation artifact: {model_evaluation_artifact}"
            )
            logging.info(
                "========== Exited initiate_model_evaluation — ModelEvaluation =========="
            )
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)