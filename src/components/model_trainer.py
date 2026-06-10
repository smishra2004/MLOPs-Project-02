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
    Trains ResNet50 for NORMAL vs PNEUMONIA classification.
    MLflow logs per-epoch metrics (loss, accuracy) during training.
    DagsHub is the remote MLflow tracking server.
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
        try:
            logging.info("Loading pretrained ResNet50.")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            for param in model.parameters():
                param.requires_grad = False

            in_features = model.fc.in_features
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
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float]:
        try:
            model.train()
            running_loss = 0.0
            correct      = 0
            total        = 0

            for images, labels in self.data_transformation_artifact.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

            return running_loss / total, correct / total

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        try:
            model.eval()
            running_loss = 0.0
            correct      = 0
            total        = 0

            with torch.no_grad():
                for images, labels in loader:
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
        MLflow logs:
          - hyperparameters once at the start (params)
          - loss and accuracy after every epoch (metrics)
        The run is NOT closed here — ModelEvaluation closes it
        after logging final test metrics.
        """
        try:
            logging.info("Starting ResNet50 training.")

            model = self.build_model()

            criterion = nn.CrossEntropyLoss(
                weight=self.data_transformation_artifact.class_weights.to(self.device)
            )
            optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.model_trainer_config.learning_rate,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

            # ── Log hyperparameters to MLflow ─────────────────────────────
            # Called once — these are the settings used for this run
            mlflow.log_params({
                "model_name"    : "resnet50",
                "num_epochs"    : self.model_trainer_config.num_epochs,
                "learning_rate" : self.model_trainer_config.learning_rate,
                "batch_size"    : self.data_transformation_artifact.train_loader.batch_size,  # ← reads from loader
                "patience"      : self.model_trainer_config.patience,
                "dropout"       : 0.4,
                "optimizer"     : "Adam",
                "loss_function" : "CrossEntropyLoss (weighted)",
                "device"        : str(self.device),
                "class_weight_normal"    : round(
                    self.data_transformation_artifact.class_weights[0].item(), 4
                ),
                "class_weight_pneumonia" : round(
                    self.data_transformation_artifact.class_weights[1].item(), 4
                ),
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

                train_loss, train_acc = self.train_one_epoch(model, criterion, optimizer)
                val_loss, val_acc     = self.evaluate(
                    model,
                    self.data_transformation_artifact.val_loader,
                    criterion,
                )
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
                # step=epoch so DagsHub plots them as a curve over time
                mlflow.log_metrics({
                    "train_loss" : round(train_loss, 4),
                    "train_acc"  : round(train_acc,  4),
                    "val_loss"   : round(val_loss,   4),
                    "val_acc"    : round(val_acc,    4),
                }, step=epoch)

                # ── Best weight checkpoint ────────────────────────────────
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

            model.load_state_dict(best_weights)
            logging.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
            return model, history

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "========== Entered initiate_model_trainer — ModelTrainer =========="
            )

            model, history = self.train()
            logging.info("Training complete.")

            criterion = nn.CrossEntropyLoss(
                weight=self.data_transformation_artifact.class_weights.to(self.device)
            )
            test_loss, test_acc = self.evaluate(
                model,
                self.data_transformation_artifact.test_loader,
                criterion,
            )
            logging.info(
                f"Test set results — "
                f"Accuracy: {test_acc:.4f}  Loss: {test_loss:.4f}"
            )

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

            model_trainer_artifact = ModelTrainerArtifact(
                model_save_path=self.model_trainer_config.model_save_path,
                test_accuracy=test_acc,
                test_loss=test_loss,
                history=history,
            )

            logging.info(f"ModelTrainer artifact: {model_trainer_artifact}")
            logging.info(
                "========== Exited initiate_model_trainer — ModelTrainer =========="
            )
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)