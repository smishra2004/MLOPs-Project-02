from aws_connection import S3Client
from src.constants import *
from dotenv import load_dotenv
load_dotenv()

s3 = S3Client()

s3.s3_client.upload_file(
    r"C:\Users\shubh\OneDrive\Desktop\chest-deep-learning.zip",
    BUCKET_NAME,
    "chest-deep-learning.zip"
)

print("Dataset uploaded")