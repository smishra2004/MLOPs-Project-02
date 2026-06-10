import sys
import torch
from PIL import Image
from torchvision import transforms

from src.entity.config_entity import ChestXrayPredictorConfig
from src.entity.chest_xray_s3_estimator import ChestXrayEstimator
from src.exception import MyException
from src.logger import logging


class ChestXrayData:
    """
    Mirrors: VehicleData

    Responsible for taking a raw uploaded image (PIL Image)
    and converting it into a normalised tensor ready for the model.

    VehicleData took form fields → DataFrame
    ChestXrayData takes PIL Image → Tensor
    """

    def __init__(self, image: Image.Image) -> None:
        """
        Args:
            image (PIL.Image.Image): Raw image uploaded by the user.
        """
        try:
            self.image = image
        except Exception as e:
            raise MyException(e, sys)

    def get_image_as_tensor(self) -> torch.Tensor:
        """
        Applies the same test/val transforms used during training:
            Resize(224, 224) → ToTensor → Normalize(ImageNet stats)

        Returns:
            torch.Tensor: shape [1, 3, 224, 224]
                          batch dimension added so model receives
                          the correct input shape
        """
        try:
            logging.info("Applying transforms to uploaded image.")

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Convert to RGB — chest X-rays are sometimes grayscale
            image_rgb    = self.image.convert('RGB')
            image_tensor = transform(image_rgb)

            # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
            image_tensor = image_tensor.unsqueeze(0)

            logging.info(f"Image tensor shape: {image_tensor.shape}")
            return image_tensor

        except Exception as e:
            raise MyException(e, sys)


class ChestXrayClassifier:
    """
    Mirrors: VehicleDataClassifier

    Loads the ResNet50 model from S3 via ChestXrayEstimator
    and runs prediction on the image tensor.

    VehicleDataClassifier.predict(dataframe) → 0 or 1
    ChestXrayClassifier.predict(image)       → 'NORMAL' or 'PNEUMONIA'
    """

    def __init__(
        self,
        prediction_pipeline_config: ChestXrayPredictorConfig = ChestXrayPredictorConfig(),
    ) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, image: Image.Image) -> str:
        """
        Full prediction flow:
            PIL Image
                ↓
            ChestXrayData.get_image_as_tensor()   apply transforms
                ↓
            ChestXrayEstimator.predict(tensor)     load model from S3 + run inference
                ↓
            'NORMAL' or 'PNEUMONIA'

        Args:
            image (PIL.Image.Image): Raw uploaded X-ray image.

        Returns:
            str: 'NORMAL' or 'PNEUMONIA'
        """
        try:
            logging.info("Entered predict method of ChestXrayClassifier.")

            # Step 1: Convert image to tensor
            chest_xray_data  = ChestXrayData(image=image)
            image_tensor     = chest_xray_data.get_image_as_tensor()

            # Step 2: Load model from S3 and predict
            estimator = ChestXrayEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            prediction = estimator.predict(image_tensor)

            # Step 3: Map prediction index to label
            # 0 = NORMAL, 1 = PNEUMONIA (matches ImageFolder class_to_idx)
            label = 'PNEUMONIA' if prediction.item() == 1 else 'NORMAL'

            logging.info(f"Prediction: {label}")
            return label

        except Exception as e:
            raise MyException(e, sys)