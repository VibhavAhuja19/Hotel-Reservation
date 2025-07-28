
import os
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.comman_function import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        # Add service account key path to config
        self.service_account_path = self.config.get("service_account_key_path", None)
        self.project_id = self.config.get("project_id", None)
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.file_name}")
    
    def _get_storage_client(self):
        """Get authenticated storage client"""
        try:
            if self.service_account_path and os.path.exists(self.service_account_path):
                # Use service account
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path
                )
                client = storage.Client(credentials=credentials, project=self.project_id)
                logger.info("Using service account for authentication")
            else:
                # Fallback to default credentials
                client = storage.Client(project=self.project_id)
                logger.info("Using default credentials for authentication")
            
            return client
        except Exception as e:
            logger.error(f"Failed to create storage client: {str(e)}")
            raise CustomException("Failed to authenticate with Google Cloud Storage", e)
    
    def download_csv_from_gcp(self):
        try:
            logger.info(f"Attempting to download {self.file_name} from bucket {self.bucket_name}")
            
            client = self._get_storage_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            # Check if blob exists
            if not blob.exists():
                raise Exception(f"File {self.file_name} does not exist in bucket {self.bucket_name}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
            
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV file is successfully downloaded to {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error(f"Error while downloading the csv file: {str(e)}")
            logger.error(f"Bucket: {self.bucket_name}, File: {self.file_name}")
            raise CustomException("Failed to download csv file", e)
        
    def split_data(self):
        try:
            logger.info("Starting the splitting process")
            
            if not os.path.exists(RAW_FILE_PATH):
                raise Exception(f"Raw data file not found at {RAW_FILE_PATH}")
            
            data = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Loaded data with shape: {data.shape}")
            
            train_data, test_data = train_test_split(
                data, 
                test_size=1-self.train_test_ratio, 
                random_state=42
            )
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(TRAIN_FILE_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(TEST_FILE_PATH), exist_ok=True)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Train data saved to {TRAIN_FILE_PATH} with shape: {train_data.shape}")
            logger.info(f"Test data saved to {TEST_FILE_PATH} with shape: {test_data.shape}")
        
        except Exception as e:
            logger.error(f"Error while splitting data: {str(e)}")
            raise CustomException("Failed to split data into training and test sets", e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise ce
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CustomException("Data ingestion failed", e)
        
        finally:
            logger.info("Data ingestion process finished")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

        

