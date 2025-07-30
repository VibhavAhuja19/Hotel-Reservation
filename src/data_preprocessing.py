import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.comman_function import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    
    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data Processing step")

            logger.info("Dropping the columns")
            df.drop(columns=['Booking_ID'] , inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoding")

            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}

            logger.info("Label Mappings are : ")
            for col,mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Doing Skewness Handling")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data", e)
        
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced Data")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_resampled , columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Data balanced sucesffuly")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e)
    
    def select_features(self,df):
        try:
            logger.info("Starting our Feature selection step")

            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                        'feature':X.columns,
                        'importance':feature_importance
                            })
            top_features_importance_df = feature_importance_df.sort_values(by="importance" , ascending=False)
            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_10_features = top_features_importance_df["feature"].head(num_features_to_select).values
            logger.info(f"Features selected : {top_10_features}")
            top_10_df = df[top_10_features.tolist() + ["booking_status"]]
            logger.info("Feature slection completed sucesfully")
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while feature selection", e)
    
    def save_data(self,df , file_path):
        try:
            logger.info("Saving our data in processed folder")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved sucesfuly to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]  

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df , PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed sucesfully")    
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)
              
    
if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()       






# import os
# import pandas as pd
# import numpy as np
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from utils.comman_function import read_yaml, load_data
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE

# logger = get_logger(__name__)

# class DataProcessor:
#     """Data processing class for handling ML data preprocessing pipeline"""
    
#     def __init__(self, train_path, test_path, processed_dir, config_path):
#         """Initialize DataProcessor with comprehensive validation and error handling"""
#         try:
#             logger.info("Initializing DataProcessor class")
            
#             # Validate input parameters
#             if not train_path or not isinstance(train_path, str):
#                 raise ValueError("train_path must be a non-empty string")
#             if not test_path or not isinstance(test_path, str):
#                 raise ValueError("test_path must be a non-empty string")
#             if not processed_dir or not isinstance(processed_dir, str):
#                 raise ValueError("processed_dir must be a non-empty string")
#             if not config_path or not isinstance(config_path, str):
#                 raise ValueError("config_path must be a non-empty string")
            
#             # Check if input files exist
#             if not os.path.exists(train_path):
#                 logger.error(f"Training data file not found: {train_path}")
#                 raise FileNotFoundError(f"Training data file not found: {train_path}")
            
#             if not os.path.exists(test_path):
#                 logger.error(f"Test data file not found: {test_path}")
#                 raise FileNotFoundError(f"Test data file not found: {test_path}")
            
#             if not os.path.exists(config_path):
#                 logger.error(f"Configuration file not found: {config_path}")
#                 raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
#             self.train_path = train_path
#             self.test_path = test_path
#             self.processed_dir = processed_dir
            
#             logger.info(f"Input paths validated - Train: {train_path}, Test: {test_path}")
#             logger.info(f"Processed directory: {processed_dir}")
            
#             # Load and validate configuration
#             try:
#                 self.config = read_yaml(config_path)
#                 logger.info(f"Configuration loaded successfully from: {config_path}")
                
#                 # Validate required configuration sections
#                 required_sections = ["data_processing"]
#                 missing_sections = [section for section in required_sections if section not in self.config]
#                 if missing_sections:
#                     raise ValueError(f"Missing required configuration sections: {missing_sections}")
                
#                 # Validate required configuration parameters
#                 data_proc_config = self.config["data_processing"]
#                 required_params = ["categorical_columns", "numerical_columns", "skewness_threshold", "no_of_features"]
#                 missing_params = [param for param in required_params if param not in data_proc_config]
#                 if missing_params:
#                     raise ValueError(f"Missing required data_processing parameters: {missing_params}")
                
#                 # Validate parameter types and values
#                 if not isinstance(data_proc_config["categorical_columns"], list):
#                     raise ValueError("categorical_columns must be a list")
#                 if not isinstance(data_proc_config["numerical_columns"], list):
#                     raise ValueError("numerical_columns must be a list")
#                 if not isinstance(data_proc_config["skewness_threshold"], (int, float)):
#                     raise ValueError("skewness_threshold must be a number")
#                 if not isinstance(data_proc_config["no_of_features"], int) or data_proc_config["no_of_features"] <= 0:
#                     raise ValueError("no_of_features must be a positive integer")
                
#                 logger.debug(f"Configuration validation passed - Parameters: {list(data_proc_config.keys())}")
                
#             except Exception as e:
#                 logger.error(f"Failed to load or validate configuration: {str(e)}")
#                 raise CustomException("Configuration loading/validation failed", e)
            
#             # Create processed directory with error handling
#             try:
#                 if not os.path.exists(self.processed_dir):
#                     os.makedirs(self.processed_dir)
#                     logger.info(f"Created processed directory: {self.processed_dir}")
#                 else:
#                     logger.info(f"Processed directory already exists: {self.processed_dir}")
                
#                 # Check if directory is writable
#                 if not os.access(self.processed_dir, os.W_OK):
#                     raise PermissionError(f"Processed directory is not writable: {self.processed_dir}")
                
#             except OSError as e:
#                 logger.error(f"Failed to create processed directory {self.processed_dir}: {str(e)}")
#                 raise CustomException(f"Cannot create processed directory: {self.processed_dir}", e)
#             except PermissionError as e:
#                 logger.error(f"Permission denied for processed directory: {str(e)}")
#                 raise CustomException(f"Permission denied for processed directory: {self.processed_dir}", e)
            
#             logger.info("DataProcessor initialization completed successfully")
            
#         except (ValueError, FileNotFoundError) as ve:
#             logger.error(f"Validation error during initialization: {str(ve)}")
#             raise CustomException(f"DataProcessor initialization failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during DataProcessor initialization: {str(e)}")
#             raise CustomException("Failed to initialize DataProcessor", e)
    
#     def preprocess_data(self, df):
#         """Preprocess data with comprehensive error handling and validation"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING DATA PREPROCESSING STEP")
#             logger.info("="*50)
            
#             # Validate input dataframe
#             if df is None:
#                 raise ValueError("Input dataframe is None")
#             if df.empty:
#                 raise ValueError("Input dataframe is empty")
            
#             logger.info(f"Input data shape: {df.shape}")
#             logger.debug(f"Input columns: {list(df.columns)}")
            
#             # Create a copy to avoid modifying original data
#             df_processed = df.copy()
            
#             # Drop Booking_ID column with validation
#             try:
#                 logger.info("Dropping the Booking_ID column")
#                 if 'Booking_ID' in df_processed.columns:
#                     df_processed.drop(columns=['Booking_ID'], inplace=True)
#                     logger.info("Booking_ID column dropped successfully")
#                 else:
#                     logger.warning("Booking_ID column not found in dataframe")
                
#             except Exception as e:
#                 logger.error(f"Error dropping Booking_ID column: {str(e)}")
#                 raise CustomException("Failed to drop Booking_ID column", e)
            
#             # Remove duplicates with validation
#             try:
#                 initial_rows = len(df_processed)
#                 df_processed.drop_duplicates(inplace=True)
#                 final_rows = len(df_processed)
#                 duplicates_removed = initial_rows - final_rows
                
#                 if duplicates_removed > 0:
#                     logger.info(f"Removed {duplicates_removed} duplicate rows")
#                 else:
#                     logger.info("No duplicate rows found")
                
#                 if df_processed.empty:
#                     raise ValueError("All rows were duplicates - resulting dataframe is empty")
                
#             except Exception as e:
#                 logger.error(f"Error removing duplicates: {str(e)}")
#                 raise CustomException("Failed to remove duplicate rows", e)
            
#             # Get column configurations
#             cat_cols = self.config["data_processing"]["categorical_columns"]
#             num_cols = self.config["data_processing"]["numerical_columns"]
            
#             logger.info(f"Categorical columns to process: {cat_cols}")
#             logger.info(f"Numerical columns to process: {num_cols}")
            
#             # Validate columns exist in dataframe
#             missing_cat_cols = [col for col in cat_cols if col not in df_processed.columns]
#             missing_num_cols = [col for col in num_cols if col not in df_processed.columns]
            
#             if missing_cat_cols:
#                 logger.error(f"Missing categorical columns: {missing_cat_cols}")
#                 raise ValueError(f"Categorical columns not found in data: {missing_cat_cols}")
            
#             if missing_num_cols:
#                 logger.error(f"Missing numerical columns: {missing_num_cols}")
#                 raise ValueError(f"Numerical columns not found in data: {missing_num_cols}")
            
#             # Apply Label Encoding with comprehensive error handling
#             try:
#                 logger.info("Starting Label Encoding process")
#                 label_encoder = LabelEncoder()
#                 mappings = {}
                
#                 for col in cat_cols:
#                     try:
#                         logger.debug(f"Encoding column: {col}")
                        
#                         # Check for missing values
#                         if df_processed[col].isnull().any():
#                             null_count = df_processed[col].isnull().sum()
#                             logger.warning(f"Column {col} has {null_count} null values - these will cause encoding to fail")
#                             raise ValueError(f"Column {col} contains null values")
                        
#                         # Check if column has any data
#                         if df_processed[col].empty:
#                             raise ValueError(f"Column {col} is empty")
                        
#                         # Get unique values count
#                         unique_count = df_processed[col].nunique()
#                         logger.debug(f"Column {col} has {unique_count} unique values")
                        
#                         # Perform encoding
#                         df_processed[col] = label_encoder.fit_transform(df_processed[col])
                        
#                         # Create mapping dictionary
#                         mappings[col] = {
#                             label: code for label, code in zip(
#                                 label_encoder.classes_, 
#                                 label_encoder.transform(label_encoder.classes_)
#                             )
#                         }
                        
#                         logger.debug(f"Successfully encoded column {col}")
                        
#                     except Exception as e:
#                         logger.error(f"Error encoding column {col}: {str(e)}")
#                         raise CustomException(f"Failed to encode categorical column: {col}", e)
                
#                 logger.info("Label Encoding completed successfully")
#                 logger.info("Label Mappings:")
#                 for col, mapping in mappings.items():
#                     logger.info(f"{col}: {mapping}")
                
#             except CustomException:
#                 raise  # Re-raise custom exceptions
#             except Exception as e:
#                 logger.error(f"Error during label encoding process: {str(e)}")
#                 raise CustomException("Label encoding process failed", e)
            
#             # Handle skewness with comprehensive error handling
#             try:
#                 logger.info("Starting skewness handling process")
#                 skew_threshold = self.config["data_processing"]["skewness_threshold"]
#                 logger.info(f"Skewness threshold: {skew_threshold}")
                
#                 # Validate numerical columns have numeric data
#                 for col in num_cols:
#                     if not pd.api.types.is_numeric_dtype(df_processed[col]):
#                         logger.warning(f"Column {col} is not numeric, skipping skewness handling")
#                         continue
                    
#                     # Check for non-positive values (log transformation requires positive values)
#                     if (df_processed[col] <= 0).any():
#                         logger.warning(f"Column {col} contains non-positive values, log transformation may cause issues")
                
#                 # Calculate skewness
#                 skewness = df_processed[num_cols].apply(lambda x: x.skew())
#                 logger.debug(f"Skewness values: {skewness.to_dict()}")
                
#                 # Identify highly skewed columns
#                 highly_skewed = skewness[skewness > skew_threshold]
                
#                 if len(highly_skewed) > 0:
#                     logger.info(f"Found {len(highly_skewed)} highly skewed columns: {list(highly_skewed.index)}")
                    
#                     for column in highly_skewed.index:
#                         try:
#                             logger.debug(f"Applying log transformation to column: {column}")
                            
#                             # Check for non-positive values before log transformation
#                             if (df_processed[column] <= 0).any():
#                                 logger.warning(f"Column {column} has non-positive values, using log1p transformation")
                            
#                             original_skew = df_processed[column].skew()
#                             df_processed[column] = np.log1p(df_processed[column])
#                             new_skew = df_processed[column].skew()
                            
#                             logger.debug(f"Column {column} - Original skewness: {original_skew:.3f}, New skewness: {new_skew:.3f}")
                            
#                         except Exception as e:
#                             logger.error(f"Error applying log transformation to column {column}: {str(e)}")
#                             raise CustomException(f"Failed to apply skewness correction to column: {column}", e)
#                 else:
#                     logger.info("No highly skewed columns found")
                
#                 logger.info("Skewness handling completed successfully")
                
#             except Exception as e:
#                 logger.error(f"Error during skewness handling: {str(e)}")
#                 raise CustomException("Skewness handling process failed", e)
            
#             logger.info(f"Preprocessing completed - Final data shape: {df_processed.shape}")
#             logger.info("="*50)
#             return df_processed
        
#         except ValueError as ve:
#             logger.error(f"Data validation error in preprocessing: {str(ve)}")
#             raise CustomException(f"Data preprocessing validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during preprocessing: {str(e)}")
#             raise CustomException("Data preprocessing failed unexpectedly", e)
        
#     def balance_data(self, df):
#         """Handle imbalanced data using SMOTE with comprehensive error handling"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING DATA BALANCING STEP")
#             logger.info("="*50)
            
#             # Validate input dataframe
#             if df is None:
#                 raise ValueError("Input dataframe is None")
#             if df.empty:
#                 raise ValueError("Input dataframe is empty")
            
#             logger.info(f"Input data shape: {df.shape}")
            
#             # Check if target column exists
#             target_col = 'booking_status'
#             if target_col not in df.columns:
#                 logger.error(f"Target column '{target_col}' not found in dataframe")
#                 raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}")
            
#             # Separate features and target
#             try:
#                 X = df.drop(columns=target_col)
#                 y = df[target_col]
                
#                 logger.info(f"Features shape: {X.shape}")
#                 logger.info(f"Target shape: {y.shape}")
                
#                 # Validate features and target
#                 if X.empty:
#                     raise ValueError("Features dataframe is empty after dropping target column")
#                 if y.empty:
#                     raise ValueError("Target series is empty")
                
#                 # Check for missing values
#                 if X.isnull().any().any():
#                     null_cols = X.columns[X.isnull().any()].tolist()
#                     logger.error(f"Features contain null values in columns: {null_cols}")
#                     raise ValueError(f"Features contain null values in columns: {null_cols}")
                
#                 if y.isnull().any():
#                     null_count = y.isnull().sum()
#                     logger.error(f"Target contains {null_count} null values")
#                     raise ValueError(f"Target contains {null_count} null values")
                
#             except Exception as e:
#                 logger.error(f"Error separating features and target: {str(e)}")
#                 raise CustomException("Failed to separate features and target", e)
            
#             # Analyze class distribution
#             try:
#                 class_counts = y.value_counts()
#                 logger.info("Original class distribution:")
#                 for class_val, count in class_counts.items():
#                     percentage = (count / len(y)) * 100
#                     logger.info(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")
                
#                 # Check if balancing is needed
#                 if len(class_counts) < 2:
#                     logger.warning("Only one class found in target variable - SMOTE cannot be applied")
#                     logger.info("Returning original dataframe without balancing")
#                     return df
                
#                 # Check minimum class size for SMOTE
#                 min_class_size = class_counts.min()
#                 if min_class_size < 2:
#                     logger.warning(f"Minimum class size is {min_class_size} - SMOTE requires at least 2 samples per class")
#                     logger.info("Returning original dataframe without balancing")
#                     return df
                
#             except Exception as e:
#                 logger.error(f"Error analyzing class distribution: {str(e)}")
#                 raise CustomException("Failed to analyze class distribution", e)
            
#             # Apply SMOTE with error handling
#             try:
#                 logger.info("Applying SMOTE for data balancing")
#                 smote = SMOTE(random_state=42)
                
#                 # Check if SMOTE can be applied
#                 logger.debug("Fitting SMOTE to the data")
#                 X_resampled, y_resampled = smote.fit_resample(X, y)
                
#                 logger.info(f"Resampled features shape: {X_resampled.shape}")
#                 logger.info(f"Resampled target shape: {y_resampled.shape}")
                
#                 # Validate resampled data
#                 if X_resampled.shape[0] == 0:
#                     raise ValueError("SMOTE resulted in empty resampled features")
#                 if len(y_resampled) == 0:
#                     raise ValueError("SMOTE resulted in empty resampled target")
                
#             except ValueError as ve:
#                 if "not enough" in str(ve).lower() or "need at least" in str(ve).lower():
#                     logger.warning(f"SMOTE failed due to insufficient data: {str(ve)}")
#                     logger.info("Returning original dataframe without balancing")
#                     return df
#                 else:
#                     logger.error(f"SMOTE validation error: {str(ve)}")
#                     raise CustomException(f"SMOTE validation failed: {str(ve)}", ve)
#             except Exception as e:
#                 logger.error(f"Error during SMOTE application: {str(e)}")
#                 raise CustomException("SMOTE application failed", e)
            
#             # Create balanced dataframe
#             try:
#                 logger.info("Creating balanced dataframe")
#                 balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
#                 balanced_df[target_col] = y_resampled
                
#                 # Validate balanced dataframe
#                 if balanced_df.empty:
#                     raise ValueError("Balanced dataframe is empty")
                
#                 # Log new class distribution
#                 new_class_counts = balanced_df[target_col].value_counts()
#                 logger.info("New class distribution after balancing:")
#                 for class_val, count in new_class_counts.items():
#                     percentage = (count / len(balanced_df)) * 100
#                     logger.info(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")
                
#                 logger.info(f"Data balancing completed successfully - Final shape: {balanced_df.shape}")
#                 logger.info("="*50)
#                 return balanced_df
                
#             except Exception as e:
#                 logger.error(f"Error creating balanced dataframe: {str(e)}")
#                 raise CustomException("Failed to create balanced dataframe", e)
        
#         except ValueError as ve:
#             logger.error(f"Data validation error in balancing: {str(ve)}")
#             raise CustomException(f"Data balancing validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during data balancing: {str(e)}")
#             raise CustomException("Data balancing failed unexpectedly", e)
    
#     def select_features(self, df):
#         """Select top features using Random Forest with comprehensive error handling"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING FEATURE SELECTION STEP")
#             logger.info("="*50)
            
#             # Validate input dataframe
#             if df is None:
#                 raise ValueError("Input dataframe is None")
#             if df.empty:
#                 raise ValueError("Input dataframe is empty")
            
#             logger.info(f"Input data shape: {df.shape}")
            
#             # Check if target column exists
#             target_col = 'booking_status'
#             if target_col not in df.columns:
#                 logger.error(f"Target column '{target_col}' not found in dataframe")
#                 raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}")
            
#             # Separate features and target
#             try:
#                 X = df.drop(columns=target_col)
#                 y = df[target_col]
                
#                 logger.info(f"Features for selection: {list(X.columns)}")
#                 logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
                
#                 # Validate features and target
#                 if X.empty:
#                     raise ValueError("Features dataframe is empty")
#                 if y.empty:
#                     raise ValueError("Target series is empty")
#                 if X.shape[1] == 0:
#                     raise ValueError("No features available for selection")
                
#                 # Check for missing values
#                 if X.isnull().any().any():
#                     null_cols = X.columns[X.isnull().any()].tolist()
#                     logger.error(f"Features contain null values in columns: {null_cols}")
#                     raise ValueError(f"Features contain null values in columns: {null_cols}")
                
#             except Exception as e:
#                 logger.error(f"Error preparing data for feature selection: {str(e)}")
#                 raise CustomException("Failed to prepare data for feature selection", e)
            
#             # Train Random Forest for feature importance
#             try:
#                 logger.info("Training Random Forest for feature importance calculation")
#                 model = RandomForestClassifier(random_state=42)
                
#                 # Check if we have enough samples for training
#                 min_samples_required = 2
#                 if len(X) < min_samples_required:
#                     raise ValueError(f"Not enough samples for training Random Forest: {len(X)} < {min_samples_required}")
                
#                 # Check class distribution
#                 unique_classes = y.nunique()
#                 if unique_classes < 2:
#                     raise ValueError(f"Need at least 2 classes for classification, found: {unique_classes}")
                
#                 logger.debug(f"Training with {len(X)} samples and {unique_classes} classes")
#                 model.fit(X, y)
#                 logger.info("Random Forest training completed successfully")
                
#             except ValueError as ve:
#                 logger.error(f"Random Forest training validation error: {str(ve)}")
#                 raise CustomException(f"Random Forest training failed: {str(ve)}", ve)
#             except Exception as e:
#                 logger.error(f"Error training Random Forest: {str(e)}")
#                 raise CustomException("Random Forest training failed", e)
            
#             # Extract and analyze feature importance
#             try:
#                 logger.info("Extracting feature importance scores")
#                 feature_importance = model.feature_importances_
                
#                 # Validate feature importance
#                 if len(feature_importance) != len(X.columns):
#                     raise ValueError(f"Feature importance length mismatch: {len(feature_importance)} != {len(X.columns)}")
                
#                 # Create feature importance dataframe
#                 feature_importance_df = pd.DataFrame({
#                     'feature': X.columns,
#                     'importance': feature_importance
#                 })
                
#                 # Sort by importance
#                 top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
                
#                 logger.info("Feature importance scores:")
#                 for idx, row in top_features_importance_df.head(10).iterrows():
#                     logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
#             except Exception as e:
#                 logger.error(f"Error extracting feature importance: {str(e)}")
#                 raise CustomException("Failed to extract feature importance", e)
            
#             # Select top features
#             try:
#                 num_features_to_select = self.config["data_processing"]["no_of_features"]
#                 logger.info(f"Selecting top {num_features_to_select} features")
                
#                 # Validate number of features to select
#                 available_features = len(X.columns)
#                 if num_features_to_select > available_features:
#                     logger.warning(f"Requested {num_features_to_select} features but only {available_features} available")
#                     num_features_to_select = available_features
                
#                 if num_features_to_select <= 0:
#                     raise ValueError(f"Number of features to select must be positive: {num_features_to_select}")
                
#                 # Get top features
#                 top_features = top_features_importance_df["feature"].head(num_features_to_select).values
#                 logger.info(f"Selected features: {list(top_features)}")
                
#                 # Create dataframe with selected features
#                 selected_columns = top_features.tolist() + [target_col]
#                 top_df = df[selected_columns]
                
#                 # Validate result
#                 if top_df.empty:
#                     raise ValueError("Feature selection resulted in empty dataframe")
                
#                 logger.info(f"Feature selection completed successfully - Final shape: {top_df.shape}")
#                 logger.info("="*50)
#                 return top_df
                
#             except KeyError as ke:
#                 logger.error(f"Column access error during feature selection: {str(ke)}")
#                 raise CustomException(f"Failed to access selected columns: {str(ke)}", ke)
#             except Exception as e:
#                 logger.error(f"Error during feature selection: {str(e)}")
#                 raise CustomException("Feature selection process failed", e)
        
#         except ValueError as ve:
#             logger.error(f"Data validation error in feature selection: {str(ve)}")
#             raise CustomException(f"Feature selection validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during feature selection: {str(e)}")
#             raise CustomException("Feature selection failed unexpectedly", e)
    
#     def save_data(self, df, file_path):
#         """Save dataframe to CSV with comprehensive error handling"""
#         try:
#             logger.info(f"Starting data save operation to: {file_path}")
            
#             # Validate input parameters
#             if df is None:
#                 raise ValueError("Input dataframe is None")
#             if df.empty:
#                 raise ValueError("Input dataframe is empty")
#             if not file_path or not isinstance(file_path, str):
#                 raise ValueError("file_path must be a non-empty string")
            
#             logger.info(f"Data to save - Shape: {df.shape}")
#             logger.debug(f"Columns: {list(df.columns)}")
            
#             # Validate file path and directory
#             try:
#                 file_dir = os.path.dirname(file_path)
                
#                 # Create directory if it doesn't exist
#                 if file_dir and not os.path.exists(file_dir):
#                     logger.info(f"Creating directory: {file_dir}")
#                     os.makedirs(file_dir, exist_ok=True)
                
#                 # Check write permissions
#                 if file_dir and not os.access(file_dir, os.W_OK):
#                     raise PermissionError(f"No write permission for directory: {file_dir}")
                
#                 # Check if file already exists and is writable
#                 if os.path.exists(file_path) and not os.access(file_path, os.W_OK):
#                     raise PermissionError(f"File exists but is not writable: {file_path}")
                
#             except OSError as e:
#                 logger.error(f"File system error: {str(e)}")
#                 raise CustomException(f"File system error for path {file_path}: {str(e)}", e)
#             except PermissionError as e:
#                 logger.error(f"Permission error: {str(e)}")
#                 raise CustomException(f"Permission denied: {str(e)}", e)
            
#             # Save data to CSV
#             try:
#                 logger.info("Writing data to CSV file")
#                 df.to_csv(file_path, index=False)
                
#                 # Verify file was created and has content
#                 if not os.path.exists(file_path):
#                     raise ValueError(f"File was not created: {file_path}")
                
#                 file_size = os.path.getsize(file_path)
#                 if file_size == 0:
#                     raise ValueError(f"Created file is empty: {file_path}")
                
#                 logger.info(f"Data saved successfully to {file_path}")
#                 logger.info(f"File size: {file_size} bytes")
                
#             except Exception as e:
#                 logger.error(f"Error writing CSV file: {str(e)}")
#                 # Clean up partial file if it exists
#                 if os.path.exists(file_path):
#                     try:
#                         os.remove(file_path)
#                         logger.debug("Cleaned up partial file")
#                     except:
#                         pass
#                 raise CustomException(f"Failed to save data to CSV: {file_path}", e)
                
#         except ValueError as ve:
#             logger.error(f"Data validation error in save_data: {str(ve)}")
#             raise CustomException(f"Data save validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during data saving: {str(e)}")
#             raise CustomException("Data saving failed unexpectedly", e)

#     def process(self):
#         """Execute the complete data processing pipeline with comprehensive error handling"""
#         try:
#             logger.info("="*60)
#             logger.info("STARTING DATA PROCESSING PIPELINE")
#             logger.info("="*60)
            
#             # Step 1: Load data from RAW directory
#             logger.info("Step 1: Loading data from RAW directory")
#             try:
#                 logger.info(f"Loading training data from: {self.train_path}")
#                 train_df = load_data(self.train_path)
                
#                 if train_df is None or train_df.empty:
#                     raise ValueError(f"Training data is empty or None: {self.train_path}")
#                 logger.info(f"Training data loaded successfully - Shape: {train_df.shape}")
                
#                 logger.info(f"Loading test data from: {self.test_path}")
#                 test_df = load_data(self.test_path)
                
#                 if test_df is None or test_df.empty:
#                     raise ValueError(f"Test data is empty or None: {self.test_path}")
#                 logger.info(f"Test data loaded successfully - Shape: {test_df.shape}")
                
#             except Exception as e:
#                 logger.error(f"Error loading raw data: {str(e)}")
#                 raise CustomException("Failed to load raw data files", e)
            
#             logger.info("Step 1 completed successfully")
            
#             # Step 2: Preprocess data
#             logger.info("Step 2: Preprocessing data")
#             try:
#                 logger.info("Preprocessing training data")
#                 train_df = self.preprocess_data(train_df)
#                 logger.info(f"Training data preprocessing completed - Shape: {train_df.shape}")
                
#                 logger.info("Preprocessing test data")
#                 test_df = self.preprocess_data(test_df)
#                 logger.info(f"Test data preprocessing completed - Shape: {test_df.shape}")
                
#             except Exception as e:
#                 logger.error(f"Error during data preprocessing: {str(e)}")
#                 raise CustomException("Data preprocessing step failed", e)
            
#             logger.info("Step 2 completed successfully")
            
#             # Step 3: Balance data
#             logger.info("Step 3: Balancing data")
#             try:
#                 logger.info("Balancing training data")
#                 train_df = self.balance_data(train_df)
#                 logger.info(f"Training data balancing completed - Shape: {train_df.shape}")
                
#                 logger.info("Balancing test data")
#                 test_df = self.balance_data(test_df)
#                 logger.info(f"Test data balancing completed - Shape: {test_df.shape}")
                
#             except Exception as e:
#                 logger.error(f"Error during data balancing: {str(e)}")
#                 raise CustomException("Data balancing step failed", e)
            
#             logger.info("Step 3 completed successfully")
            
#             # Step 4: Feature selection
#             logger.info("Step 4: Feature selection")
#             try:
#                 logger.info("Selecting features from training data")
#                 train_df = self.select_features(train_df)
#                 logger.info(f"Training data feature selection completed - Shape: {train_df.shape}")
                
#                 # Apply same feature selection to test data
#                 logger.info("Applying same feature selection to test data")
#                 if list(train_df.columns) != list(test_df.columns):
#                     # Get selected columns from training data
#                     selected_columns = list(train_df.columns)
                    
#                     # Check if all selected columns exist in test data
#                     missing_cols = [col for col in selected_columns if col not in test_df.columns]
#                     if missing_cols:
#                         logger.error(f"Selected columns missing in test data: {missing_cols}")
#                         raise ValueError(f"Selected columns not found in test data: {missing_cols}")
                    
#                     # Apply same column selection to test data
#                     test_df = test_df[selected_columns]
#                     logger.info(f"Test data feature selection completed - Shape: {test_df.shape}")
#                 else:
#                     logger.info("Test data already has the same columns as training data")
                
#             except Exception as e:
#                 logger.error(f"Error during feature selection: {str(e)}")
#                 raise CustomException("Feature selection step failed", e)
            
#             logger.info("Step 4 completed successfully")
            
#             # Step 5: Save processed data
#             logger.info("Step 5: Saving processed data")
#             try:
#                 logger.info("Saving processed training data")
#                 self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
#                 logger.info(f"Training data saved to: {PROCESSED_TRAIN_DATA_PATH}")
                
#                 logger.info("Saving processed test data")
#                 self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
#                 logger.info(f"Test data saved to: {PROCESSED_TEST_DATA_PATH}")
                
#             except Exception as e:
#                 logger.error(f"Error saving processed data: {str(e)}")
#                 raise CustomException("Data saving step failed", e)
            
#             logger.info("Step 5 completed successfully")
            
#             # Pipeline completion summary
#             logger.info("="*60)
#             logger.info("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
#             logger.info("="*60)
#             logger.info(f"Final training data shape: {train_df.shape}")
#             logger.info(f"Final test data shape: {test_df.shape}")
#             logger.info(f"Final columns: {list(train_df.columns)}")
#             logger.info("="*60)
            
#         except CustomException as ce:
#             logger.error("="*60)
#             logger.error(f"DATA PROCESSING PIPELINE FAILED - CustomException: {str(ce)}")
#             logger.error("="*60)
#             raise ce
        
#         except Exception as e:
#             logger.error("="*60)
#             logger.error(f"DATA PROCESSING PIPELINE FAILED - Unexpected error: {str(e)}")
#             logger.error("="*60)
#             raise CustomException("Data processing pipeline failed unexpectedly", e)
        
#         finally:
#             logger.info("Data processing pipeline execution finished")

# if __name__ == "__main__":
#     try:
#         logger.info("Starting data processing script")
        
#         # Validate required paths exist
#         required_paths = [TRAIN_FILE_PATH, TEST_FILE_PATH, CONFIG_PATH]
#         path_names = ["TRAIN_FILE_PATH", "TEST_FILE_PATH", "CONFIG_PATH"]
        
#         for path, name in zip(required_paths, path_names):
#             if not os.path.exists(path):
#                 logger.error(f"Required path does not exist - {name}: {path}")
#                 raise FileNotFoundError(f"Required file not found - {name}: {path}")
        
#         logger.info("All required paths validated successfully")
        
#         # Initialize and run data processor
#         processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
#         processor.process()
        
#         logger.info("Data processing script completed successfully")
        
#     except CustomException as ce:
#         logger.error(f"Script failed with CustomException: {str(ce)}")
#         exit(1)
    
#     except FileNotFoundError as fe:
#         logger.error(f"Script failed - Required file not found: {str(fe)}")
#         exit(1)
    
#     except Exception as e:
#         logger.error(f"Script failed with unexpected error: {str(e)}")
#         raise CustomException("Data processing script failed", e)