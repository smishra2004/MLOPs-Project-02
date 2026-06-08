print("DEMO STARTED")

from src.pipeline.training_pipeline import TrainPipeline

print("PIPELINE IMPORTED")

pipeline = TrainPipeline()

print("PIPELINE CREATED")

pipeline.run_pipeline()

print("PIPELINE FINISHED")