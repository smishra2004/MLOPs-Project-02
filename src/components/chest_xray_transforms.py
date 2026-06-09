import sys
from torchvision import transforms
from src.exception import MyException
from src.logger import logging


class ChestXrayTransforms:
    """
    Defines train and test/val transforms for the chest X-ray dataset.

    Mirrors: ChestXrayData — responsible for one thing only:
    building and returning the transform objects.

    Key decisions from EDA:
    - Resize to 224x224   : 199 unique sizes found
    - ImageNet normalize  : standard for pretrained models (ResNet50)
    - Augment train only  : medically realistic augmentations only
    - No augment test/val : we need clean, unmodified images for evaluation
    """

    def __init__(self) -> None:
        try:
            # ── Target size for all images (ResNet50 standard) ────────────
            self.img_size = 224

            # ── ImageNet stats (used because we'll load a pretrained model) 
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]

        except Exception as e:
            raise MyException(e, sys)

    def get_train_transforms(self) -> transforms.Compose:
        """
        Returns augmented transforms for the TRAIN split only.

        Augmentations chosen based on EDA + medical realism:
        - RandomHorizontalFlip : chest can appear mirrored
        - RandomRotation(10)   : slight patient positioning variation
        - ColorJitter           : brightness variation across X-ray machines
        - RandomResizedCrop    : slight zoom variation in imaging distance

        NOT used (medically unrealistic):
        - Vertical flip        : upside-down chest has no meaning
        - Large rotations      : X-rays are always roughly upright
        - Heavy distortions    : would destroy diagnostic features
        """
        try:
            logging.info("Building train transforms.")
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.9, 1.0)   # max 10% zoom — subtle, medically safe
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            logging.info("Train transforms built successfully.")
            return train_transforms

        except Exception as e:
            raise MyException(e, sys)

    def get_test_val_transforms(self) -> transforms.Compose:
        """
        Returns clean transforms for TEST and VAL splits.

        No augmentation — we need the raw image for honest evaluation.
        Only resize + normalize so the model receives the same format
        it was trained on.
        """
        try:
            logging.info("Building test/val transforms.")
            test_val_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            logging.info("Test/val transforms built successfully.")
            return test_val_transforms

        except Exception as e:
            raise MyException(e, sys)