import os
import sys

import mlflow
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


@dataclass
class EvaluateModelResponse:
    """
    Holds the full comparison between newly trained model
    and current production model in S3.
    """
    trained_model_auc       : float
    trained_model_accuracy  : float
    trained_model_loss      : float
    trained_model_precision : float
    trained_model_recall    : float
    trained_model_f1        : float
    best_model_auc          : float   # 0.0 if no production model exists yet
    best_model_accuracy     : float   # 0.0 if no production model exists yet
    is_model_accepted       : bool
    difference              : float   # trained_auc - best_auc


class ModelEvaluation:
    """
    Single responsibility: EVALUATE the trained model on the test set
    and decide whether it beats the current production model.

    What this file does:
        - loads saved .pth from ModelTrainerArtifact
        - runs on test_loader ONCE to compute all metrics
        - checks S3 for production model and compares AUC
        - logs all metrics + confusion matrix to MLflow/DagsHub
        - returns ModelEvaluationArtifact with acceptance decision

    What this file does NOT do:
        - never trains anything
        - never touches train_loader or val_loader
        - never opens/closes MLflow run (that is training_pipeline.py's job)
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
    def _load_trained_model(self) -> nn.Module:
        """
        Loads the locally saved .pth file from ModelTrainerArtifact.
        Rebuilds the same architecture used during training.
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
            logging.info("Trained model loaded successfully.")
            return model

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def get_best_model(self) -> Optional[ChestXrayEstimator]:
        """
        Checks S3 for a production model.
        Returns ChestXrayEstimator if found, None on first deployment.
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
            logging.info("No production model in S3 — first deployment.")
            return None

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_all_metrics(
        self,
        model : nn.Module,
    ) -> dict:
        """
        Runs model on test_loader and computes all metrics.
        This is the ONLY place test_loader is used in the entire pipeline.

        Metrics:
            loss      : weighted CrossEntropyLoss
            accuracy  : correct / total
            auc       : ROC-AUC (main comparison metric — threshold independent)
            precision : of predicted PNEUMONIA, how many were actually PNEUMONIA
            recall    : of actual PNEUMONIA, how many did we correctly catch
                        ← most critical metric for medical diagnosis
            f1        : harmonic mean of precision and recall
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
            all_probs    = []   # P(PNEUMONIA) — needed for AUC

            with torch.no_grad():
                for images, labels in self.data_transformation_artifact.test_loader:
                    images  = images.to(self.device)
                    labels  = labels.to(self.device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    running_loss += loss.item() * images.size(0)
                    total        += labels.size(0)

                    probs    = torch.softmax(outputs, dim=1)[:, 1]
                    _, preds = torch.max(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            metrics = {
                "loss"      : round(running_loss / total,                          4),
                "accuracy"  : round(accuracy_score(all_labels, all_preds),         4),
                "auc"       : round(roc_auc_score(all_labels, all_probs),          4),
                "precision" : round(precision_score(all_labels, all_preds,
                                    zero_division=0),                              4),
                "recall"    : round(recall_score(all_labels, all_preds,
                                    zero_division=0),                              4),
                "f1"        : round(f1_score(all_labels, all_preds,
                                    zero_division=0),                              4),
                "_preds"    : all_preds,
                "_labels"   : all_labels,
            }
            return metrics

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _save_confusion_matrix(self, preds: list, labels: list) -> str:
        """
        Saves confusion matrix as PNG and returns path.
        Logged to MLflow as an artifact so it appears on DagsHub.
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
        Core evaluation logic — only thing this function does is evaluate.

        Steps:
            1. Load trained model from local .pth
            2. Run on test_loader → compute all metrics
            3. Log trained model metrics to MLflow
            4. Log training history summary to MLflow
            5. Save + log confusion matrix
            6. Log model file as artifact
            7. Check S3 for production model
            8. If found → evaluate it on same test set → compare AUC
            9. Make acceptance decision
        """
        try:
            logging.info("Starting model evaluation on test set.")

            # ── Step 1 + 2: Load trained model → compute metrics ──────────
            trained_model   = self._load_trained_model()
            trained_metrics = self._compute_all_metrics(model=trained_model)

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

            # ── Step 3: Log test metrics to MLflow ────────────────────────
            mlflow.log_metrics({
                "test_loss"      : trained_metrics['loss'],
                "test_accuracy"  : trained_metrics['accuracy'],
                "test_auc"       : trained_metrics['auc'],
                "test_precision" : trained_metrics['precision'],
                "test_recall"    : trained_metrics['recall'],
                "test_f1"        : trained_metrics['f1'],
            })

            # ── Step 4: Log training history summary to MLflow ────────────
            history = self.model_trainer_artifact.history
            if history['train_loss']:
                mlflow.log_metrics({
                    "final_train_loss" : round(history['train_loss'][-1], 4),
                    "final_train_acc"  : round(history['train_acc'][-1],  4),
                    "final_val_loss"   : round(history['val_loss'][-1],   4),
                    "final_val_acc"    : round(history['val_acc'][-1],    4),
                })

            # ── Step 5: Save + log confusion matrix ───────────────────────
            cm_path = self._save_confusion_matrix(
                preds=trained_metrics['_preds'],
                labels=trained_metrics['_labels'],
            )
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

            # ── Step 6: Log model file to MLflow ──────────────────────────
            mlflow.log_artifact(
                self.model_trainer_artifact.model_save_path,
                artifact_path="model"
            )

            # ── Step 7 + 8: Check production model in S3 ──────────────────
            best_model_auc      = 0.0
            best_model_accuracy = 0.0
            best_model          = self.get_best_model()

            if best_model is not None:
                logging.info("Evaluating production model from S3 on test set.")
                production_model   = best_model.load_model()
                production_metrics = self._compute_all_metrics(model=production_model)
                best_model_auc     = production_metrics['auc']
                best_model_accuracy= production_metrics['accuracy']

                logging.info(
                    f"Production model — "
                    f"AUC: {best_model_auc}  Acc: {best_model_accuracy}"
                )
                mlflow.log_metrics({
                    "production_model_auc"      : best_model_auc,
                    "production_model_accuracy" : best_model_accuracy,
                })
            else:
                logging.info(
                    "No production model in S3 — trained model accepted by default."
                )

            # ── Step 9: Acceptance decision ───────────────────────────────
            is_model_accepted = trained_metrics['auc'] > best_model_auc
            difference        = trained_metrics['auc'] - best_model_auc

            mlflow.log_metrics({
                "auc_improvement" : round(difference, 4),
                "model_accepted"  : int(is_model_accepted),
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
        Entry point. Calls evaluate_model() and packages the artifact.
        MLflow run is already open from training_pipeline.py — no start/end here.
        """
        try:
            logging.info(
                "========== Entered initiate_model_evaluation — ModelEvaluation =========="
            )

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted  = evaluate_model_response.is_model_accepted,
                s3_model_path      = self.model_eval_config.s3_model_key_path,
                trained_model_path = self.model_trainer_artifact.model_save_path,
                changed_accuracy   = evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info(
                "========== Exited initiate_model_evaluation — ModelEvaluation =========="
            )
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)