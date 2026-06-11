import os
import sys
import copy
import time

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.exception import MyException
from src.logger import logging


class ModelTrainer:
    """
    Single responsibility: TRAIN the ResNet50 model and save weights.

    What this file does:
        - builds ResNet50 with frozen backbone + custom head
        - trains on train_loader
        - validates on val_loader after each epoch
        - early stopping based on val_loss
        - saves best weights to .pth
        - logs hyperparams + per-epoch metrics to MLflow

    What this file does NOT do:
        - never touches test_loader  (that is ModelEvaluation's job)
        - never computes test metrics (that is ModelEvaluation's job)
        - never opens/closes MLflow run (that is training_pipeline.py's job)
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ) -> None:
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config         = model_trainer_config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"ModelTrainer initialised. Device: {self.device}")
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        """
        Loads pretrained ResNet50, freezes backbone,
        attaches 2-class head for NORMAL vs PNEUMONIA.
        """
        try:
            logging.info("Loading pretrained ResNet50.")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Freeze all 49 backbone layers — only head will train
            for param in model.parameters():
                param.requires_grad = False

            # Replace final FC with our 2-class head
            in_features = model.fc.in_features          # 2048
            model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, 2)
            )

            model = model.to(self.device)

            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(
                f"ResNet50 built — "
                f"trainable: {trainable_params:,} / {total_params:,} params "
                f"({100 * trainable_params / total_params:.1f}%)"
            )
            return model

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def train_one_epoch(
        self,
        model    : nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float]:
        """
        One full pass over train_loader.
        Weights are updated here — this is the only place weights change.

        Returns:
            avg_loss (float), accuracy (float)
        """
        try:
            model.train()
            running_loss = 0.0
            correct      = 0
            total        = 0

            for images, labels in self.data_transformation_artifact.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()           # clear old gradients
                outputs = model(images)         # forward pass
                loss    = criterion(outputs, labels)
                loss.backward()                 # compute gradients
                optimizer.step()               # update weights

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

            return running_loss / total, correct / total

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_val(
        self,
        model    : nn.Module,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """
        Evaluates on val_loader only — used during training loop
        to track improvement and make early stopping decisions.

        NO weight updates happen here.
        Returns: avg_loss (float), accuracy (float)
        """
        try:
            model.eval()
            running_loss = 0.0
            correct      = 0
            total        = 0

            with torch.no_grad():
                for images, labels in self.data_transformation_artifact.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    correct  += (preds == labels).sum().item()
                    total    += labels.size(0)

            return running_loss / total, correct / total

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def train(self) -> tuple[nn.Module, dict]:
        """
        Full training loop.

        Uses val_loader to:
            - decide when to save best weights (checkpoint)
            - decide when to stop early (patience)
            - decide when to reduce learning rate (scheduler)


        MLflow logs:
            - hyperparameters once at start   → mlflow.log_params()
            - loss + acc after every epoch    → mlflow.log_metrics(step=epoch)

        Returns:
            model   : ResNet50 with best val checkpoint loaded
            history : dict of train/val loss + acc per epoch
        """
        try:
            logging.info("Starting ResNet50 training.")

            model = self.build_model()

            # Weighted loss — handles 2.89x class imbalance from EDA
            criterion = nn.CrossEntropyLoss(
                weight=self.data_transformation_artifact.class_weights.to(self.device)
            )

            # Adam only on trainable head parameters
            optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.model_trainer_config.learning_rate,
            )

            # Halve LR if val_loss doesn't improve for 2 epochs
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

            # ── Log hyperparameters once ──────────────────────────────────
            mlflow.log_params({
                "model_name"             : "resnet50",
                "num_epochs"             : self.model_trainer_config.num_epochs,
                "learning_rate"          : self.model_trainer_config.learning_rate,
                "batch_size"             : self.data_transformation_artifact.train_loader.batch_size,
                "patience"               : self.model_trainer_config.patience,
                "dropout"                : 0.4,
                "optimizer"              : "Adam",
                "loss_function"          : "CrossEntropyLoss (weighted)",
                "device"                 : str(self.device),
                "class_weight_normal"    : round(self.data_transformation_artifact.class_weights[0].item(), 4),
                "class_weight_pneumonia" : round(self.data_transformation_artifact.class_weights[1].item(), 4),
            })

            history = {
                'train_loss': [],
                'train_acc' : [],
                'val_loss'  : [],
                'val_acc'   : [],
            }

            best_val_loss = float('inf')
            best_weights  = None
            patience_ctr  = 0

            for epoch in range(self.model_trainer_config.num_epochs):
                epoch_start = time.time()

                # ── Train on train_loader ─────────────────────────────────
                train_loss, train_acc = self.train_one_epoch(
                    model, criterion, optimizer
                )

                # ── Validate on val_loader ────────────────────────────────
                val_loss, val_acc = self.evaluate_val(model, criterion)

                scheduler.step(val_loss)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                epoch_time = time.time() - epoch_start
                logging.info(
                    f"Epoch [{epoch + 1}/{self.model_trainer_config.num_epochs}]  "
                    f"Train -> Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                    f"Val -> Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
                    f"[{epoch_time:.0f}s]"
                )

                # ── Log per-epoch metrics to MLflow ───────────────────────
                mlflow.log_metrics({
                    "train_loss" : round(train_loss, 4),
                    "train_acc"  : round(train_acc,  4),
                    "val_loss"   : round(val_loss,   4),
                    "val_acc"    : round(val_acc,    4),
                }, step=epoch)

                # ── Best checkpoint + early stopping ──────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights  = copy.deepcopy(model.state_dict())
                    patience_ctr  = 0
                    logging.info(
                        f"Val loss improved to {best_val_loss:.4f} — checkpoint saved."
                    )
                else:
                    patience_ctr += 1
                    logging.info(
                        f"Val loss did not improve. "
                        f"Patience: {patience_ctr}/{self.model_trainer_config.patience}"
                    )
                    if patience_ctr >= self.model_trainer_config.patience:
                        logging.info(
                            f"Early stopping triggered at epoch {epoch + 1}."
                        )
                        break

            # Restore best weights before returning
            model.load_state_dict(best_weights)
            logging.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
            return model, history

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Entry point. Only trains and saves — nothing else.

        Steps:
            1. train()              → train on train, validate on val
            2. torch.save()         → save best weights to .pth
            3. ModelTrainerArtifact → pass model path + history to ModelEvaluation

        ModelEvaluation will handle everything related to test set.
        """
        try:
            logging.info(
                "========== Entered initiate_model_trainer — ModelTrainer =========="
            )

            # Step 1: Train
            model, history = self.train()
            logging.info("Training complete.")

            # Step 2: Save best weights to disk
            os.makedirs(
                os.path.dirname(self.model_trainer_config.model_save_path),
                exist_ok=True,
            )
            torch.save(
                model.state_dict(),
                self.model_trainer_config.model_save_path,
            )
            logging.info(
                f"Model saved to: {self.model_trainer_config.model_save_path}"
            )

            # Step 3: Return artifact — no test metrics, those belong in ModelEvaluation
            model_trainer_artifact = ModelTrainerArtifact(
                model_save_path = self.model_trainer_config.model_save_path,
                history         = history,
            )

            logging.info(f"ModelTrainer artifact: {model_trainer_artifact}")
            logging.info(
                "========== Exited initiate_model_trainer — ModelTrainer =========="
            )
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)