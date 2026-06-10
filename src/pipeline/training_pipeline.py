import sys
import mlflow
import dagshub
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.chest_xray_transforms import ChestXrayTransforms
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from dotenv import load_dotenv
load_dotenv()

from src.entity.config_entity import (DataIngestionConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig,
                                      ModelPusherConfig)

from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ModelEvaluationArtifact,
                                        ModelPusherArtifact)
print("TRAINING_PIPELINE IMPORTED")

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config      = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config       = ModelTrainerConfig()
        self.model_evaluation_config    = ModelEvaluationConfig()
        self.model_pusher_config        = ModelPusherConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        print("START_DATA_INGESTION CALLED")
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from aws S3")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the zipped data from aws test train and val")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                    )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)

    def start_model_evaluation(self,model_eval_config : ModelEvaluationConfig,
                               data_transformation_artifact: DataTransformationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_transformation_artifact=data_transformation_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys)
    
    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        print("RUN_PIPELINE CALLED")
        try:
            dagshub.init(repo_owner='smishra2004', repo_name='MLOPs-Project-02', mlflow=True)
            mlflow.set_tracking_uri("https://dagshub.com/smishra2004/MLOPs-Project-02.mlflow")
            mlflow.set_experiment("chest-xray-detection-experiment")
            
            with mlflow.start_run() as run:
                mlflow.set_tag("pipeline_status", "running")
                data_ingestion_artifact      = self.start_data_ingestion()
                data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
                model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
                model_evaluation_artifact = self.start_model_evaluation(model_eval_config=self.model_evaluation_config,
                                                                        data_transformation_artifact=data_transformation_artifact,
                                                                        model_trainer_artifact=model_trainer_artifact)
                if not model_evaluation_artifact.is_model_accepted:
                    logging.info(f"Model not accepted.")
                    return None
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                
                mlflow.set_tag("pipeline_status", "completed")
            
        except Exception as e:
            mlflow.set_tag("pipeline_status", "failed")
            raise MyException(e, sys)