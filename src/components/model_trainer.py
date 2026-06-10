import os
import sys
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import mlflow
import mlflow.pytorch

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.exception import MyException
from src.logger import logging

from dotenv import load_dotenv
load_dotenv()


class ModelTrainer:
    """
    Trains a ResNet50 model (pretrained on ImageNet) for binary
    classification: NORMAL vs PNEUMONIA.

    Transfer learning strategy:
      - Backbone (all ResNet50 layers) → frozen
      - Custom head (Linear → ReLU → Dropout → Linear) → trainable only

    Why freeze the backbone?
      ResNet50 pretrained on ImageNet already knows edges, textures,
      and shapes. With only 5216 training images, training the full
      network would overfit. We only teach the head NORMAL vs PNEUMONIA.
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
        Loads pretrained ResNet50, freezes all backbone layers,
        replaces the final FC layer with a 2-class head.

        Architecture of the trainable head:
            Linear(2048 → 256) → ReLU → Dropout(0.4) → Linear(256 → 2)

        Dropout(0.4): regularisation — reduces overfitting on small dataset.
        256 hidden units: enough capacity without being too large for CPU.
        """
        try:
            mlflow.pytorch.autolog(log_models=True, log_every_n_epoch=1)
            logging.info("Loading pretrained ResNet50.")
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Freeze all backbone layers
            for param in model.parameters():
                param.requires_grad = False

            # Replace final FC layer with custom 2-class head
            in_features = model.fc.in_features   # 2048 for ResNet50
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
        """
        Runs one full pass over the training DataLoader.

        Returns:
            avg_loss (float): mean loss across all batches
            accuracy (float): fraction of correct predictions
        """
        try:
            model.train()
            running_loss = 0.0
            correct      = 0
            total        = 0

            for batch_idx, (images, labels) in enumerate(
                self.data_transformation_artifact.train_loader
            ):
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

            avg_loss = running_loss / total
            accuracy = correct / total
            return avg_loss, accuracy

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """
        Evaluates the model on a given DataLoader (val or test).
        No gradients computed — inference only.

        Returns:
            avg_loss (float): mean loss
            accuracy (float): fraction of correct predictions
        """
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

            avg_loss = running_loss / total
            accuracy = correct / total
            return avg_loss, accuracy

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def train(self) -> tuple[nn.Module, dict]:
        """
        Full training loop with:
          - Weighted CrossEntropyLoss  : handles 2.89x class imbalance
          - Adam optimiser             : only on trainable head params
          - ReduceLROnPlateau          : halves LR if val loss stalls
          - Early stopping             : stops if val loss doesn't improve
                                         for `patience` epochs
          - Best weight tracking       : always restores best val checkpoint

        Returns:
            model   : ResNet50 with best validation weights loaded
            history : dict of per-epoch train/val loss and accuracy
        """
        try:
            logging.info("Starting ResNet50 training.")

            model = self.build_model()

            # Weighted loss — class_weights from DataTransformationArtifact
            # Higher weight on NORMAL (minority) so misclassifying it costs more
            criterion = nn.CrossEntropyLoss(
                weight=self.data_transformation_artifact.class_weights.to(self.device)
            )

            # Adam on trainable params only (the head)
            optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.model_trainer_config.learning_rate,
            )

            # Reduce LR by 0.5 if val loss doesn't improve for 2 epochs
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )

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

                # ── Train ────────────────────────────────────────────────
                train_loss, train_acc = self.train_one_epoch(
                    model, criterion, optimizer
                )

                # ── Validate ─────────────────────────────────────────────
                val_loss, val_acc = self.evaluate(
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

            # Restore best checkpoint before returning
            model.load_state_dict(best_weights)
            logging.info(
                f"Training complete. Best val loss: {best_val_loss:.4f}"
            )
            return model, history

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Entry point for the model trainer component.
        Mirrors: initiate_data_ingestion() / initiate_data_transformation()

        Steps:
          1. Train ResNet50 with best-weight tracking
          2. Evaluate on test set
          3. Save model weights to artifact path
          4. Return ModelTrainerArtifact

        Returns:
            ModelTrainerArtifact: save path + test metrics + history
        """
        try:
            logging.info(
                "========== Entered initiate_model_trainer — ModelTrainer =========="
            )

            # Step 1: Train
            model, history = self.train()
            logging.info("Training complete.")

            # Step 2: Evaluate on test set
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

            # Step 3: Save model weights
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

            # Step 4: Build and return artifact
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