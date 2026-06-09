from multiprocessing import freeze_support
from src.pipeline.training_pipeline import TrainPipeline


def main():
    pipeline = TrainPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    freeze_support()
    main()