
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

        

# import os
# import pandas as pd
# from google.cloud import storage
# from google.oauth2 import service_account
# from sklearn.model_selection import train_test_split
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from utils.comman_function import read_yaml

# logger = get_logger(__name__)

# class DataIngestion:
#     def __init__(self, config):
#         """Initialize DataIngestion with configuration parameters"""
#         try:
#             logger.info("Initializing DataIngestion class")
            
#             # Validate config structure
#             if not config or "data_ingestion" not in config:
#                 raise ValueError("Invalid configuration: 'data_ingestion' section missing")
            
#             self.config = config["data_ingestion"]
            
#             # Validate required configuration parameters
#             required_params = ["bucket_name", "bucket_file_name", "train_ratio"]
#             missing_params = [param for param in required_params if param not in self.config]
#             if missing_params:
#                 raise ValueError(f"Missing required configuration parameters: {missing_params}")
            
#             self.bucket_name = self.config["bucket_name"]
#             self.file_name = self.config["bucket_file_name"]
#             self.train_test_ratio = self.config["train_ratio"]
            
#             # Validate train_test_ratio
#             if not (0 < self.train_test_ratio < 1):
#                 raise ValueError(f"train_ratio must be between 0 and 1, got: {self.train_test_ratio}")
            
#             # Add service account key path to config
#             self.service_account_path = self.config.get("service_account_key_path", None)
#             self.project_id = self.config.get("project_id", None)
            
#             # Create raw directory with proper error handling
#             try:
#                 os.makedirs(RAW_DIR, exist_ok=True)
#                 logger.info(f"Raw directory ensured at: {RAW_DIR}")
#             except OSError as e:
#                 logger.error(f"Failed to create raw directory {RAW_DIR}: {str(e)}")
#                 raise CustomException(f"Failed to create raw directory: {RAW_DIR}", e)
            
#             logger.info(f"DataIngestion initialized successfully - Bucket: {self.bucket_name}, File: {self.file_name}, Train ratio: {self.train_test_ratio}")
            
#         except ValueError as ve:
#             logger.error(f"Configuration validation error: {str(ve)}")
#             raise CustomException(f"Invalid configuration provided: {str(ve)}", ve)
#         except Exception as e:
#             logger.error(f"Unexpected error during DataIngestion initialization: {str(e)}")
#             raise CustomException("Failed to initialize DataIngestion", e)
    
#     def _get_storage_client(self):
#         """Get authenticated storage client with comprehensive error handling"""
#         try:
#             logger.info("Attempting to create Google Cloud Storage client")
            
#             if self.service_account_path:
#                 logger.debug(f"Checking service account path: {self.service_account_path}")
                
#                 if not os.path.exists(self.service_account_path):
#                     logger.warning(f"Service account file not found at: {self.service_account_path}")
#                     raise FileNotFoundError(f"Service account file not found: {self.service_account_path}")
                
#                 if not os.access(self.service_account_path, os.R_OK):
#                     logger.error(f"Service account file is not readable: {self.service_account_path}")
#                     raise PermissionError(f"Cannot read service account file: {self.service_account_path}")
                
#                 try:
#                     # Use service account
#                     credentials = service_account.Credentials.from_service_account_file(
#                         self.service_account_path
#                     )
#                     client = storage.Client(credentials=credentials, project=self.project_id)
#                     logger.info("Successfully authenticated using service account")
                    
#                 except Exception as e:
#                     logger.error(f"Failed to load service account credentials: {str(e)}")
#                     raise CustomException("Invalid service account credentials", e)
#             else:
#                 # Fallback to default credentials
#                 logger.info("No service account path provided, using default credentials")
#                 try:
#                     client = storage.Client(project=self.project_id)
#                     logger.info("Successfully authenticated using default credentials")
#                 except Exception as e:
#                     logger.error(f"Failed to authenticate with default credentials: {str(e)}")
#                     raise CustomException("Default credentials authentication failed", e)
            
#             # Test client connectivity
#             try:
#                 # Test if we can access the project
#                 client.get_service_account_email()
#                 logger.debug("Storage client connectivity test passed")
#             except Exception as e:
#                 logger.warning(f"Storage client connectivity test failed: {str(e)}")
#                 # Continue anyway as this might still work for bucket operations
            
#             return client
            
#         except (FileNotFoundError, PermissionError) as fe:
#             logger.error(f"File access error: {str(fe)}")
#             raise CustomException(f"Service account file access error: {str(fe)}", fe)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error creating storage client: {str(e)}")
#             raise CustomException("Failed to authenticate with Google Cloud Storage", e)
    
#     def download_csv_from_gcp(self):
#         """Download CSV file from Google Cloud Storage with enhanced error handling"""
#         try:
#             logger.info(f"Starting download process - Bucket: {self.bucket_name}, File: {self.file_name}")
            
#             # Validate inputs
#             if not self.bucket_name or not self.bucket_name.strip():
#                 raise ValueError("Bucket name cannot be empty")
#             if not self.file_name or not self.file_name.strip():
#                 raise ValueError("File name cannot be empty")
            
#             # Get storage client
#             client = self._get_storage_client()
            
#             # Get bucket with error handling
#             try:
#                 bucket = client.bucket(self.bucket_name)
#                 logger.debug(f"Successfully accessed bucket: {self.bucket_name}")
#             except Exception as e:
#                 logger.error(f"Failed to access bucket {self.bucket_name}: {str(e)}")
#                 raise CustomException(f"Cannot access bucket: {self.bucket_name}", e)
            
#             # Get blob with error handling
#             try:
#                 blob = bucket.blob(self.file_name)
#                 logger.debug(f"Created blob reference for file: {self.file_name}")
#             except Exception as e:
#                 logger.error(f"Failed to create blob reference for {self.file_name}: {str(e)}")
#                 raise CustomException(f"Cannot create reference to file: {self.file_name}", e)
            
#             # Check if blob exists
#             try:
#                 if not blob.exists():
#                     logger.error(f"File {self.file_name} does not exist in bucket {self.bucket_name}")
#                     raise FileNotFoundError(f"File {self.file_name} not found in bucket {self.bucket_name}")
#                 logger.debug(f"Confirmed file exists: {self.file_name}")
#             except Exception as e:
#                 if isinstance(e, FileNotFoundError):
#                     raise CustomException(str(e), e)
#                 logger.error(f"Error checking if file exists: {str(e)}")
#                 raise CustomException(f"Cannot verify file existence: {self.file_name}", e)
            
#             # Create directory if it doesn't exist
#             try:
#                 raw_dir = os.path.dirname(RAW_FILE_PATH)
#                 os.makedirs(raw_dir, exist_ok=True)
#                 logger.debug(f"Ensured directory exists: {raw_dir}")
#             except OSError as e:
#                 logger.error(f"Failed to create directory {raw_dir}: {str(e)}")
#                 raise CustomException(f"Cannot create directory for raw file: {raw_dir}", e)
            
#             # Download file with progress tracking
#             try:
#                 logger.info(f"Starting file download to: {RAW_FILE_PATH}")
#                 file_size = blob.size
#                 logger.debug(f"File size: {file_size} bytes")
                
#                 blob.download_to_filename(RAW_FILE_PATH)
                
#                 # Verify download
#                 if not os.path.exists(RAW_FILE_PATH):
#                     raise FileNotFoundError(f"Downloaded file not found at: {RAW_FILE_PATH}")
                
#                 downloaded_size = os.path.getsize(RAW_FILE_PATH)
#                 logger.info(f"File downloaded successfully - Size: {downloaded_size} bytes, Path: {RAW_FILE_PATH}")
                
#                 if file_size and downloaded_size != file_size:
#                     logger.warning(f"Size mismatch - Expected: {file_size}, Downloaded: {downloaded_size}")
                
#             except Exception as e:
#                 logger.error(f"Error during file download: {str(e)}")
#                 # Clean up partial download
#                 if os.path.exists(RAW_FILE_PATH):
#                     try:
#                         os.remove(RAW_FILE_PATH)
#                         logger.debug("Cleaned up partial download")
#                     except:
#                         pass
#                 raise CustomException(f"Failed to download file from GCS: {self.file_name}", e)
            
#         except ValueError as ve:
#             logger.error(f"Input validation error: {str(ve)}")
#             raise CustomException(f"Invalid input parameters: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during CSV download: {str(e)}")
#             logger.error(f"Context - Bucket: {self.bucket_name}, File: {self.file_name}")
#             raise CustomException("Failed to download CSV file from GCP", e)
        
#     def split_data(self):
#         """Split data into training and test sets with enhanced error handling"""
#         try:
#             logger.info(f"Starting data splitting process with train ratio: {self.train_test_ratio}")
            
#             # Verify raw file exists
#             if not os.path.exists(RAW_FILE_PATH):
#                 logger.error(f"Raw data file not found at: {RAW_FILE_PATH}")
#                 raise FileNotFoundError(f"Raw data file not found: {RAW_FILE_PATH}")
            
#             # Check file size and readability
#             try:
#                 file_size = os.path.getsize(RAW_FILE_PATH)
#                 logger.debug(f"Raw file size: {file_size} bytes")
                
#                 if file_size == 0:
#                     raise ValueError("Raw data file is empty")
                
#                 if not os.access(RAW_FILE_PATH, os.R_OK):
#                     raise PermissionError(f"Cannot read raw data file: {RAW_FILE_PATH}")
                    
#             except OSError as e:
#                 logger.error(f"Error accessing raw file: {str(e)}")
#                 raise CustomException(f"Cannot access raw data file: {RAW_FILE_PATH}", e)
            
#             # Load data with comprehensive error handling
#             try:
#                 logger.info(f"Loading data from: {RAW_FILE_PATH}")
#                 data = pd.read_csv(RAW_FILE_PATH)
#                 logger.info(f"Successfully loaded data with shape: {data.shape}")
                
#                 # Validate loaded data
#                 if data.empty:
#                     raise ValueError("Loaded dataset is empty")
                
#                 if len(data) < 2:
#                     raise ValueError(f"Dataset too small for splitting: {len(data)} rows")
                
#                 logger.debug(f"Data validation passed - Rows: {len(data)}, Columns: {len(data.columns)}")
                
#             except pd.errors.EmptyDataError:
#                 logger.error("CSV file is empty or has no data")
#                 raise CustomException("CSV file contains no data", None)
#             except pd.errors.ParserError as pe:
#                 logger.error(f"CSV parsing error: {str(pe)}")
#                 raise CustomException("Failed to parse CSV file - file may be corrupted", pe)
#             except Exception as e:
#                 logger.error(f"Error loading CSV data: {str(e)}")
#                 raise CustomException("Failed to load data from CSV file", e)
            
#             # Perform train-test split with error handling
#             try:
#                 logger.info(f"Splitting data - Train ratio: {self.train_test_ratio}")
#                 test_size = 1 - self.train_test_ratio
                
#                 train_data, test_data = train_test_split(
#                     data, 
#                     test_size=test_size, 
#                     random_state=42
#                 )
                
#                 logger.info(f"Data split completed - Train: {train_data.shape}, Test: {test_data.shape}")
                
#                 # Validate split results
#                 if train_data.empty or test_data.empty:
#                     raise ValueError("Data split resulted in empty datasets")
                
#             except Exception as e:
#                 logger.error(f"Error during train-test split: {str(e)}")
#                 raise CustomException("Failed to split data into train and test sets", e)
            
#             # Create directories and save files
#             try:
#                 # Create directories if they don't exist
#                 train_dir = os.path.dirname(TRAIN_FILE_PATH)
#                 test_dir = os.path.dirname(TEST_FILE_PATH)
                
#                 os.makedirs(train_dir, exist_ok=True)
#                 os.makedirs(test_dir, exist_ok=True)
#                 logger.debug(f"Ensured directories exist - Train: {train_dir}, Test: {test_dir}")
                
#                 # Save training data
#                 logger.info(f"Saving training data to: {TRAIN_FILE_PATH}")
#                 train_data.to_csv(TRAIN_FILE_PATH, index=False)
                
#                 # Verify training file
#                 if not os.path.exists(TRAIN_FILE_PATH) or os.path.getsize(TRAIN_FILE_PATH) == 0:
#                     raise ValueError("Failed to save training data properly")
                
#                 # Save test data
#                 logger.info(f"Saving test data to: {TEST_FILE_PATH}")
#                 test_data.to_csv(TEST_FILE_PATH, index=False)
                
#                 # Verify test file
#                 if not os.path.exists(TEST_FILE_PATH) or os.path.getsize(TEST_FILE_PATH) == 0:
#                     raise ValueError("Failed to save test data properly")
                
#                 logger.info(f"Data splitting completed successfully")
#                 logger.info(f"Train data: {TRAIN_FILE_PATH} - Shape: {train_data.shape}")
#                 logger.info(f"Test data: {TEST_FILE_PATH} - Shape: {test_data.shape}")
                
#             except OSError as e:
#                 logger.error(f"File system error during data saving: {str(e)}")
#                 raise CustomException("Failed to save split data files", e)
#             except Exception as e:
#                 logger.error(f"Error saving split data: {str(e)}")
#                 raise CustomException("Failed to save training and test data", e)
        
#         except ValueError as ve:
#             logger.error(f"Data validation error: {str(ve)}")
#             raise CustomException(f"Data validation failed: {str(ve)}", ve)
#         except FileNotFoundError as fe:
#             logger.error(f"File not found error: {str(fe)}")
#             raise CustomException(str(fe), fe)
#         except PermissionError as pe:
#             logger.error(f"Permission error: {str(pe)}")
#             raise CustomException(f"File access permission denied: {str(pe)}", pe)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during data splitting: {str(e)}")
#             raise CustomException("Failed to split data into training and test sets", e)
        
#     def run(self):
#         """Execute the complete data ingestion pipeline with comprehensive error handling"""
#         try:
#             logger.info("="*60)
#             logger.info("STARTING DATA INGESTION PROCESS")
#             logger.info("="*60)
            
#             # Step 1: Download CSV from GCP
#             logger.info("Step 1: Downloading CSV from GCP")
#             self.download_csv_from_gcp()
#             logger.info("Step 1 completed successfully")
            
#             # Step 2: Split data
#             logger.info("Step 2: Splitting data into train and test sets")
#             self.split_data()
#             logger.info("Step 2 completed successfully")
            
#             logger.info("="*60)
#             logger.info("DATA INGESTION PROCESS COMPLETED SUCCESSFULLY")
#             logger.info("="*60)
        
#         except CustomException as ce:
#             logger.error("="*60)
#             logger.error(f"DATA INGESTION FAILED - CustomException: {str(ce)}")
#             logger.error("="*60)
#             raise ce
        
#         except Exception as e:
#             logger.error("="*60)
#             logger.error(f"DATA INGESTION FAILED - Unexpected error: {str(e)}")
#             logger.error("="*60)
#             raise CustomException("Data ingestion process failed unexpectedly", e)
        
#         finally:
#             logger.info("Data ingestion process finished - cleaning up resources")

# if __name__ == "__main__":
#     try:
#         logger.info("Starting data ingestion script")
        
#         # Load configuration with error handling
#         try:
#             config = read_yaml(CONFIG_PATH)
#             logger.info(f"Configuration loaded successfully from: {CONFIG_PATH}")
#         except Exception as e:
#             logger.error(f"Failed to load configuration from {CONFIG_PATH}: {str(e)}")
#             raise CustomException("Configuration loading failed", e)
        
#         # Initialize and run data ingestion
#         data_ingestion = DataIngestion(config)
#         data_ingestion.run()
        
#         logger.info("Data ingestion script completed successfully")
        
#     except CustomException as ce:
#         logger.error(f"Script failed with CustomException: {str(ce)}")
#         exit(1)
#     except Exception as e:
#         logger.error(f"Script failed with unexpected error: {str(e)}")
#         raise CustomException("Data ingestion script failed", e)