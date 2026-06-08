import os
from datetime import date

# For MongoDB connection
DATABASE_NAME = "Proj2"
COLLECTION_NAME = "Proj2-Data"
MONGODB_URL_KEY = "MONGODB_URL"

#AWS connection
S3_BUCKET_NAME = "chest-x-ray-dataset-cnn"
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"
S3_DATA_KEY = "chest-deep-learning.zip"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifacts"