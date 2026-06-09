import os
import sys
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.exception import MyException
from src.logger import logging
from src.components.chest_xray_transforms import ChestXrayTransforms


class DataTransformation:
    """
    Applies transforms to the ingested dataset and builds DataLoaders
    ready for model training.

    Mirrors: DataIngestion — calls the data access layer (ChestXrayTransforms),
    does its own work on top (builds datasets + dataloaders), returns an artifact.

    Responsibilities:
    - Apply train augmentation transforms to train split
    - Apply clean transforms to test and val splits
    - Compute class weights from EDA imbalance finding (ratio 2.89x)
    - Build and return DataLoaders for all three splits
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ) -> None:
        try:
            self.data_ingestion_artifact    = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MyException(e, sys)

    def get_class_weights(self, train_dataset: datasets.ImageFolder) -> torch.Tensor:
        """
        Computes class weights to handle the 2.89x imbalance found in EDA.
        Higher weight = more penalty for misclassifying that class.

        Formula: weight[class] = total_samples / (n_classes * count[class])
        This is the standard sklearn-style balanced class weight formula.

        Args:
            train_dataset: The training ImageFolder dataset.

        Returns:
            torch.Tensor: [weight_NORMAL, weight_PNEUMONIA]
        """
        try:
            logging.info("Computing class weights for imbalanced dataset.")

            class_counts = np.bincount(train_dataset.targets)
            total        = sum(class_counts)
            n_classes    = len(class_counts)

            # weight = total / (n_classes * count)  — mirrors sklearn's 'balanced'
            weights = torch.tensor(
                [total / (n_classes * count) for count in class_counts],
                dtype=torch.float
            )

            for cls_name, idx in train_dataset.class_to_idx.items():
                logging.info(
                    f"Class weight [{cls_name}]: {weights[idx]:.4f}  "
                    f"(count={class_counts[idx]})"
                )

            return weights

        except Exception as e:
            raise MyException(e, sys)

    def build_datasets(
        self,
    ) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
        """
        Builds ImageFolder datasets for all three splits using the
        transforms from ChestXrayTransforms.

        ImageFolder automatically reads class labels from folder names:
            train/NORMAL/     → label 0
            train/PNEUMONIA/  → label 1

        Returns:
            Tuple of (train_dataset, test_dataset, val_dataset)
        """
        try:
            logging.info("Building ImageFolder datasets.")
            xray_transforms = ChestXrayTransforms()

            train_dataset = datasets.ImageFolder(
                root=self.data_ingestion_artifact.train_dir,
                transform=xray_transforms.get_train_transforms(),
            )
            test_dataset = datasets.ImageFolder(
                root=self.data_ingestion_artifact.test_dir,
                transform=xray_transforms.get_test_val_transforms(),
            )
            val_dataset = datasets.ImageFolder(
                root=self.data_ingestion_artifact.val_dir,
                transform=xray_transforms.get_test_val_transforms(),
            )

            logging.info(
                f"Datasets built — "
                f"train: {len(train_dataset)}  "
                f"test: {len(test_dataset)}  "
                f"val: {len(val_dataset)}"
            )
            logging.info(f"Class mapping: {train_dataset.class_to_idx}")

            return train_dataset, test_dataset, val_dataset

        except Exception as e:
            raise MyException(e, sys)

    def build_dataloaders(
        self,
        train_dataset: datasets.ImageFolder,
        test_dataset:  datasets.ImageFolder,
        val_dataset:   datasets.ImageFolder,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Wraps datasets into DataLoaders.

        Train DataLoader uses WeightedRandomSampler so the model sees
        a balanced mix of NORMAL and PNEUMONIA in every batch — this
        directly addresses the 2.89x imbalance from EDA.

        Test and val DataLoaders use no sampler — sequential, unshuffled,
        clean evaluation.

        Args:
            train_dataset, test_dataset, val_dataset: ImageFolder datasets.

        Returns:
            Tuple of (train_loader, test_loader, val_loader)
        """
        try:
            logging.info("Building DataLoaders.")

            # ── Weighted sampler for train — handles 2.89x imbalance ──────
            class_weights  = self.get_class_weights(train_dataset)
            sample_weights = [
                class_weights[label] for label in train_dataset.targets
            ]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.data_transformation_config.batch_size,
                sampler=sampler,           # replaces shuffle=True when using sampler
                num_workers=self.data_transformation_config.num_workers,
                pin_memory=torch.cuda.is_available(),           # speeds up CPU → GPU transfer
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.data_transformation_config.batch_size,
                shuffle=False,
                num_workers=self.data_transformation_config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.data_transformation_config.batch_size,
                shuffle=False,
                num_workers=self.data_transformation_config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            logging.info(
                f"DataLoaders built — "
                f"train batches: {len(train_loader)}  "
                f"test batches: {len(test_loader)}  "
                f"val batches: {len(val_loader)}"
            )

            return train_loader, test_loader, val_loader

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Entry point for the data transformation component.
        Mirrors: initiate_data_ingestion()

        Returns:
            DataTransformationArtifact: DataLoaders + class weights
            ready to be consumed by the model trainer.
        """
        try:
            logging.info(
                "========== Entered initiate_data_transformation — DataTransformation =========="
            )

            # Step 1: Build datasets with transforms applied
            train_dataset, test_dataset, val_dataset = self.build_datasets()
            logging.info("Datasets with transforms ready.")

            # Step 2: Build dataloaders with weighted sampler for train
            train_loader, test_loader, val_loader = self.build_dataloaders(
                train_dataset, test_dataset, val_dataset
            )
            logging.info("DataLoaders ready.")

            # Step 3: Compute class weights for loss function
            class_weights = self.get_class_weights(train_dataset)
            logging.info(f"Class weights for loss function: {class_weights.tolist()}")

            # Step 4: Build and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                train_loader=train_loader,
                test_loader=test_loader,
                val_loader=val_loader,
                class_weights=class_weights,
            )

            logging.info(f"Data transformation artifact created.")
            logging.info(
                "========== Exited initiate_data_transformation — DataTransformation =========="
            )

            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)