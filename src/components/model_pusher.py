import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.chest_xray_s3_estimator import ChestXrayEstimator
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import MyException
from src.logger import logging


class ModelPusher:
    """
    Mirrors: ModelPusher from reference project.

    Pushes the newly trained model to S3 only if it was accepted
    by ModelEvaluation (i.e. AUC beats the current production model).

    If model is not accepted → logs the reason and skips the upload.
    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config      : ModelPusherConfig,
    ) -> None:
        """
        Args:
            model_evaluation_artifact : Output from ModelEvaluation stage.
            model_pusher_config       : Bucket name + S3 key for the model.
        """
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config       = model_pusher_config
            self.s3                        = SimpleStorageService()
            self.chest_xray_estimator      = ChestXrayEstimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path,
            )
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Uploads the accepted model to S3.
        Mirrors: ModelPusher.initiate_model_pusher()

        Flow:
          1. Check model_evaluation_artifact.is_model_accepted
          2. If accepted → upload local .pth to S3 via ChestXrayEstimator
          3. Return ModelPusherArtifact with bucket + S3 path
          4. If not accepted → log reason, return artifact with is_pushed=False

        Returns:
            ModelPusherArtifact: bucket name, S3 path, and push status.
        """
        try:
            logging.info(
                "========== Entered initiate_model_pusher — ModelPusher =========="
            )

            if not self.model_evaluation_artifact.is_model_accepted:
                # Model did not beat production — do not push
                logging.info(
                    f"Model not accepted by evaluator — skipping S3 upload. "
                    f"AUC improvement: {self.model_evaluation_artifact.changed_accuracy:.4f}"
                )
                model_pusher_artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path,
                    is_pushed=False,
                )
                logging.info(f"Model pusher artifact: {model_pusher_artifact}")
                return model_pusher_artifact

            # Model accepted — upload to S3
            logging.info(
                f"Model accepted. AUC improvement: "
                f"{self.model_evaluation_artifact.changed_accuracy:.4f}. "
                f"Uploading to S3."
            )

            self.chest_xray_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path,
                remove=False,   # keep local copy, same as reference
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
                is_pushed=True,
            )

            logging.info(f"Model pushed to S3 successfully.")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info(
                "========== Exited initiate_model_pusher — ModelPusher =========="
            )
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)
