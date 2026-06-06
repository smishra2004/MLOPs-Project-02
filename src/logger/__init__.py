import logging
import os
from datetime import datetime
from from_root import from_root
from logging.handlers import RotatingFileHandler

#constants
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

#log directory folder
log_dir_path = os.path.join(from_root(),LOG_DIR)
os.makedirs(log_dir_path,exist_ok=True)
#log file path for file handler
log_file_path = os.path.join(log_dir_path,LOG_FILE)

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    #defining format of logging
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    
    #terminal handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    #file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

configure_logger()
# Suppress noisy third-party debug logs
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
logging.getLogger("pymongo.connection").setLevel(logging.WARNING)
logging.getLogger("pymongo.serverSelection").setLevel(logging.WARNING)
logging.getLogger("pymongo.command").setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
    