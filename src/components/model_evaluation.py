import os
import sys
import tempfile

import mlflow
import mlflow.pytorch
import dagshub
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torchvision import models

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.entity.chest_xray_s3_estimator import ChestXrayEstimator
from src.exception import MyException
from src.logger import logging
from src.constants import DAGSHUB_USERNAME, DAGSHUB_REPO_NAME


@dataclass
class EvaluateModelResponse:
    """
    Full metrics comparison between newly trained model
    and the current production model in S3.
    """
    # newly trained model metrics
    trained_model_auc       : float
    trained_model_accuracy  : float
    trained_model_loss      : float
    trained_model_precision : float
    trained_model_recall    : float
    trained_model_f1        : float

    # production model metrics (all 0.0 if no production model exists)
    best_model_auc          : float
    best_model_accuracy     : float

    # decision
    is_model_accepted       : bool
    difference              : float     # trained_auc - best_auc


class ModelEvaluation:
    """
    Evaluates the trained model against the production model in S3.
    Logs ALL metrics to MLflow / DagsHub.

    MLflow run lifecycle:
        ModelEvaluation OPENS  the run  → mlflow.start_run()
        ModelEvaluation CLOSES the run  → mlflow.end_run()
    ModelTrainer only logs inside an already-open run (no start/end there).
    """

    def __init__(
        self,
        model_eval_config            : ModelEvaluationConfig,
        data_transformation_artifact : DataTransformationArtifact,
        model_trainer_artifact       : ModelTrainerArtifact,
    ) -> None:
        try:
            self.model_eval_config            = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact       = model_trainer_artifact
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _setup_mlflow(self) -> None:
        """
        Connects MLflow to DagsHub as the remote tracking server.
        Call once before mlflow.start_run().

        Fill in your DagsHub credentials in constants/__init__.py:
            DAGSHUB_USERNAME  = "your_dagshub_username"
            DAGSHUB_REPO_NAME = "your_repo_name"
        """
        try:
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True,
            )
            mlflow.set_experiment("ChestXray_ResNet50")
            logging.info("MLflow connected to DagsHub successfully.")
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _load_trained_model(self) -> nn.Module:
        """
        Loads the locally saved ResNet50 from the ModelTrainer artifact path.
        Rebuilds the same architecture used during training then pours in weights.
        """
        try:
            logging.info(
                f"Loading trained model from: "
                f"{self.model_trainer_artifact.model_save_path}"
            )
            model       = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc    = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 2),
            )
            model.load_state_dict(
                torch.load(
                    self.model_trainer_artifact.model_save_path,
                    map_location=self.device,
                )
            )
            model = model.to(self.device)
            model.eval()
            logging.info("Trained model loaded.")
            return model
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def get_best_model(self) -> Optional[ChestXrayEstimator]:
        """
        Fetches the current production model from S3 if it exists.
        Returns None on first deployment.
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
    def _compute_all_metrics(
        self,
        model    : nn.Module,
        loader   : torch.utils.data.DataLoader,
    ) -> dict:
        """
        Computes the full set of evaluation metrics for a given model.

        Metrics computed:
            loss      : weighted CrossEntropyLoss (same as training)
            accuracy  : fraction of correct predictions
            auc       : ROC-AUC score (main comparison metric)
            precision : of all predicted PNEUMONIA, how many were correct
            recall    : of all actual PNEUMONIA, how many did we catch
                        (most important for medical — missing pneumonia = bad)
            f1        : harmonic mean of precision and recall

        Args:
            model  : PyTorch model in eval mode.
            loader : DataLoader (test set).

        Returns:
            dict of all metric values.
        """
        try:
            model.eval()
            criterion = nn.CrossEntropyLoss(
                weight=self.data_transformation_artifact.class_weights.to(self.device)
            )

            running_loss = 0.0
            total        = 0
            all_preds    = []
            all_labels   = []
            all_probs    = []   # P(PNEUMONIA) for AUC

            with torch.no_grad():
                for images, labels in loader:
                    images  = images.to(self.device)
                    labels  = labels.to(self.device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    running_loss += loss.item() * images.size(0)
                    total        += labels.size(0)

                    probs        = torch.softmax(outputs, dim=1)[:, 1]
                    _, preds     = torch.max(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            metrics = {
                "loss"      : round(running_loss / total,                              4),
                "accuracy"  : round(accuracy_score(all_labels, all_preds),             4),
                "auc"       : round(roc_auc_score(all_labels, all_probs),              4),
                "precision" : round(precision_score(all_labels, all_preds,
                                    zero_division=0),                                  4),
                "recall"    : round(recall_score(all_labels, all_preds,
                                    zero_division=0),                                  4),
                "f1"        : round(f1_score(all_labels, all_preds,
                                    zero_division=0),                                  4),
            }

            # Store raw arrays for confusion matrix + classification report
            metrics['_preds']  = all_preds
            metrics['_labels'] = all_labels

            return metrics

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _save_confusion_matrix(self, preds: list, labels: list) -> str:
        """
        Saves a confusion matrix plot as a PNG and returns the file path.
        We log this image to MLflow as an artifact.
        """
        try:
            cm  = confusion_matrix(labels, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_title('Confusion Matrix — Test Set', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['NORMAL', 'PNEUMONIA'])
            ax.set_yticklabels(['NORMAL', 'PNEUMONIA'])
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=14, fontweight='bold'
                    )
            plt.colorbar(im)
            plt.tight_layout()

            save_path = os.path.join(
                os.path.dirname(self.model_trainer_artifact.model_save_path),
                'confusion_matrix.png'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close()
            logging.info(f"Confusion matrix saved: {save_path}")
            return save_path

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Core evaluation logic:
          1. Compute full metrics for trained model on test set
          2. Check if production model exists in S3
          3. If yes → compute metrics for production model too
          4. Compare AUC to decide acceptance
          5. Log everything to MLflow

        Returns:
            EvaluateModelResponse with all metrics + acceptance decision.
        """
        try:
            logging.info("Starting model evaluation.")

            # ── Step 1: Compute trained model metrics ─────────────────────
            trained_model   = self._load_trained_model()
            trained_metrics = self._compute_all_metrics(
                model=trained_model,
                loader=self.data_transformation_artifact.test_loader,
            )

            logging.info(
                f"Trained model metrics — "
                f"AUC: {trained_metrics['auc']}  "
                f"Acc: {trained_metrics['accuracy']}  "
                f"Loss: {trained_metrics['loss']}  "
                f"Precision: {trained_metrics['precision']}  "
                f"Recall: {trained_metrics['recall']}  "
                f"F1: {trained_metrics['f1']}"
            )
            logging.info(
                "\n" + classification_report(
                    trained_metrics['_labels'],
                    trained_metrics['_preds'],
                    target_names=['NORMAL', 'PNEUMONIA'],
                )
            )

            # ── Step 2: Log trained model metrics to MLflow ───────────────
            mlflow.log_metrics({
                "test_loss"      : trained_metrics['loss'],
                "test_accuracy"  : trained_metrics['accuracy'],
                "test_auc"       : trained_metrics['auc'],
                "test_precision" : trained_metrics['precision'],
                "test_recall"    : trained_metrics['recall'],
                "test_f1"        : trained_metrics['f1'],
            })

            # ── Step 3: Log training history metrics from ModelTrainer ─────
            # These were logged per epoch in model_trainer.py already
            # We additionally log final epoch values as summary here
            history = self.model_trainer_artifact.history
            if history['train_loss']:
                mlflow.log_metrics({
                    "final_train_loss" : round(history['train_loss'][-1], 4),
                    "final_train_acc"  : round(history['train_acc'][-1],  4),
                    "final_val_loss"   : round(history['val_loss'][-1],   4),
                    "final_val_acc"    : round(history['val_acc'][-1],    4),
                })

            # ── Step 4: Log confusion matrix image to MLflow ──────────────
            cm_path = self._save_confusion_matrix(
                preds=trained_metrics['_preds'],
                labels=trained_metrics['_labels'],
            )
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

            # ── Step 5: Log the model file itself to MLflow ───────────────
            mlflow.log_artifact(
                self.model_trainer_artifact.model_save_path,
                artifact_path="model"
            )

            # ── Step 6: Check production model ────────────────────────────
            best_model_auc      = 0.0
            best_model_accuracy = 0.0
            best_model          = self.get_best_model()

            if best_model is not None:
                logging.info("Computing metrics for production model from S3.")
                production_model    = best_model.load_model()
                production_metrics  = self._compute_all_metrics(
                    model=production_model,
                    loader=self.data_transformation_artifact.test_loader,
                )
                best_model_auc      = production_metrics['auc']
                best_model_accuracy = production_metrics['accuracy']

                logging.info(
                    f"Production model metrics — "
                    f"AUC: {best_model_auc}  "
                    f"Acc: {best_model_accuracy}"
                )

                # Log production model metrics for comparison
                mlflow.log_metrics({
                    "production_model_auc"      : best_model_auc,
                    "production_model_accuracy" : best_model_accuracy,
                })
            else:
                logging.info(
                    "No production model in S3 — "
                    "trained model accepted by default."
                )

            # ── Step 7: Acceptance decision ───────────────────────────────
            is_model_accepted = trained_metrics['auc'] > best_model_auc
            difference        = trained_metrics['auc'] - best_model_auc

            mlflow.log_metrics({
                "auc_improvement"  : round(difference, 4),
                "model_accepted"   : int(is_model_accepted),   # 1=yes, 0=no
            })

            result = EvaluateModelResponse(
                trained_model_auc       = trained_metrics['auc'],
                trained_model_accuracy  = trained_metrics['accuracy'],
                trained_model_loss      = trained_metrics['loss'],
                trained_model_precision = trained_metrics['precision'],
                trained_model_recall    = trained_metrics['recall'],
                trained_model_f1        = trained_metrics['f1'],
                best_model_auc          = best_model_auc,
                best_model_accuracy     = best_model_accuracy,
                is_model_accepted       = is_model_accepted,
                difference              = difference,
            )

            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Entry point for the model evaluation component.
        Opens AND closes the MLflow run here.
        ModelTrainer only logs metrics inside the already-open run.
        """
        try:
            logging.info(
                "========== Entered initiate_model_evaluation — ModelEvaluation =========="
            )

            # ── Connect to DagsHub ────────────────────────────────────────
            

            # ── Open MLflow run ───────────────────────────────────────────
            # Everything logged in ModelTrainer + ModelEvaluation
            # goes into this single run
            

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted  = evaluate_model_response.is_model_accepted,
                    s3_model_path      = self.model_eval_config.s3_model_key_path,
                    trained_model_path = self.model_trainer_artifact.model_save_path,
                    changed_accuracy   = evaluate_model_response.difference,
            )

            logging.info(
                    f"Model evaluation artifact: {model_evaluation_artifact}"
            )

            # run closes here — all metrics are now on DagsHub
            logging.info("MLflow run closed. All metrics logged to DagsHub.")
            logging.info(
                "========== Exited initiate_model_evaluation — ModelEvaluation =========="
            )
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)