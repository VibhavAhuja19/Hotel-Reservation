import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.comman_function import read_yaml,load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data splitted sucefully for Model Training")

            return X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data" ,  e)
        
    def train_lgbm(self,X_train,y_train):
        try:
            logger.info("Intializing our model")

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting our Hyperparamter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            logger.info("Starting our Hyperparamter tuning")

            random_search.fit(X_train,y_train)

            logger.info("Hyperparamter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best paramters are : {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model" ,  e)
    
    def evaluate_model(self , model , X_test , y_test):
        try:
            logger.info("Evaluating our model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")

            return {
                "accuracy" : accuracy,
                "precison" : precision,
                "recall" : recall,
                "f1" : f1
            }
        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model" ,  e)
        
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("saving the model")
            joblib.dump(model , self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model" ,  e)
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our Model Training pipeline")

                logger.info("Starting our MLFLOW experimentation")

                logger.info("Logging the training and testing datset to MLFLOW")
                mlflow.log_artifact(self.train_path , artifact_path="datasets")
                mlflow.log_artifact(self.test_path , artifact_path="datasets")

                X_train,y_train,X_test,y_test =self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgbm_model ,X_test , y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and metrics to MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training sucesfullly completed")

        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline" ,  e)
        
if __name__=="__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()
        

# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import RandomizedSearchCV
# import lightgbm as lgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from config.model_params import *
# from utils.comman_function import read_yaml, load_data
# from scipy.stats import randint

# import mlflow
# import mlflow.sklearn

# logger = get_logger(__name__)

# class ModelTraining:
#     """Model training class for LightGBM with MLflow integration"""
    
#     def __init__(self, train_path, test_path, model_output_path):
#         """Initialize ModelTraining with comprehensive validation and error handling"""
#         try:
#             logger.info("Initializing ModelTraining class")
            
#             # Validate input parameters
#             if not train_path or not isinstance(train_path, str):
#                 raise ValueError("train_path must be a non-empty string")
#             if not test_path or not isinstance(test_path, str):
#                 raise ValueError("test_path must be a non-empty string")
#             if not model_output_path or not isinstance(model_output_path, str):
#                 raise ValueError("model_output_path must be a non-empty string")
            
#             # Check if input files exist
#             if not os.path.exists(train_path):
#                 logger.error(f"Training data file not found: {train_path}")
#                 raise FileNotFoundError(f"Training data file not found: {train_path}")
            
#             if not os.path.exists(test_path):
#                 logger.error(f"Test data file not found: {test_path}")
#                 raise FileNotFoundError(f"Test data file not found: {test_path}")
            
#             # Check file permissions
#             if not os.access(train_path, os.R_OK):
#                 raise PermissionError(f"Cannot read training data file: {train_path}")
            
#             if not os.access(test_path, os.R_OK):
#                 raise PermissionError(f"Cannot read test data file: {test_path}")
            
#             # Validate model output directory
#             model_dir = os.path.dirname(model_output_path)
#             if model_dir and not os.path.exists(model_dir):
#                 try:
#                     os.makedirs(model_dir, exist_ok=True)
#                     logger.info(f"Created model output directory: {model_dir}")
#                 except OSError as e:
#                     logger.error(f"Failed to create model directory {model_dir}: {str(e)}")
#                     raise CustomException(f"Cannot create model output directory: {model_dir}", e)
            
#             # Check write permissions for model directory
#             if model_dir and not os.access(model_dir, os.W_OK):
#                 raise PermissionError(f"No write permission for model directory: {model_dir}")
            
#             self.train_path = train_path
#             self.test_path = test_path
#             self.model_output_path = model_output_path
            
#             logger.info(f"Paths validated - Train: {train_path}")
#             logger.info(f"Test: {test_path}")
#             logger.info(f"Model output: {model_output_path}")
            
#             # Validate and load model parameters
#             try:
#                 # Validate LIGHTGM_PARAMS exists and is a dictionary
#                 if 'LIGHTGM_PARAMS' not in globals():
#                     raise ValueError("LIGHTGM_PARAMS not found in model_params")
#                 if not isinstance(LIGHTGM_PARAMS, dict):
#                     raise ValueError("LIGHTGM_PARAMS must be a dictionary")
                
#                 self.params_dist = LIGHTGM_PARAMS
#                 logger.info(f"LightGBM parameters loaded: {list(self.params_dist.keys())}")
                
#                 # Validate RANDOM_SEARCH_PARAMS exists and is a dictionary
#                 if 'RANDOM_SEARCH_PARAMS' not in globals():
#                     raise ValueError("RANDOM_SEARCH_PARAMS not found in model_params")
#                 if not isinstance(RANDOM_SEARCH_PARAMS, dict):
#                     raise ValueError("RANDOM_SEARCH_PARAMS must be a dictionary")
                
#                 # Validate required random search parameters
#                 required_params = ["n_iter", "cv", "n_jobs", "verbose", "random_state", "scoring"]
#                 missing_params = [param for param in required_params if param not in RANDOM_SEARCH_PARAMS]
#                 if missing_params:
#                     raise ValueError(f"Missing required random search parameters: {missing_params}")
                
#                 self.random_search_params = RANDOM_SEARCH_PARAMS
#                 logger.info(f"Random search parameters loaded: {self.random_search_params}")
                
#                 # Validate parameter values
#                 if self.random_search_params["n_iter"] <= 0:
#                     raise ValueError("n_iter must be positive")
#                 if self.random_search_params["cv"] <= 1:
#                     raise ValueError("cv must be greater than 1")
                
#             except Exception as e:
#                 logger.error(f"Error loading model parameters: {str(e)}")
#                 raise CustomException("Failed to load model parameters", e)
            
#             logger.info("ModelTraining initialization completed successfully")
            
#         except (ValueError, FileNotFoundError, PermissionError) as ve:
#             logger.error(f"Validation error during initialization: {str(ve)}")
#             raise CustomException(f"ModelTraining initialization failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during ModelTraining initialization: {str(e)}")
#             raise CustomException("Failed to initialize ModelTraining", e)

#     def load_and_split_data(self):
#         """Load and split data with comprehensive error handling and validation"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING DATA LOADING AND SPLITTING")
#             logger.info("="*50)
            
#             # Load training data
#             try:
#                 logger.info(f"Loading training data from: {self.train_path}")
#                 train_df = load_data(self.train_path)
                
#                 if train_df is None:
#                     raise ValueError("Training data is None")
#                 if train_df.empty:
#                     raise ValueError("Training data is empty")
                
#                 logger.info(f"Training data loaded successfully - Shape: {train_df.shape}")
#                 logger.debug(f"Training data columns: {list(train_df.columns)}")
                
#             except Exception as e:
#                 logger.error(f"Error loading training data: {str(e)}")
#                 raise CustomException(f"Failed to load training data from {self.train_path}", e)
            
#             # Load test data
#             try:
#                 logger.info(f"Loading test data from: {self.test_path}")
#                 test_df = load_data(self.test_path)
                
#                 if test_df is None:
#                     raise ValueError("Test data is None")
#                 if test_df.empty:
#                     raise ValueError("Test data is empty")
                
#                 logger.info(f"Test data loaded successfully - Shape: {test_df.shape}")
#                 logger.debug(f"Test data columns: {list(test_df.columns)}")
                
#             except Exception as e:
#                 logger.error(f"Error loading test data: {str(e)}")
#                 raise CustomException(f"Failed to load test data from {self.test_path}", e)
            
#             # Validate data consistency
#             try:
#                 # Check if target column exists
#                 target_col = "booking_status"
#                 if target_col not in train_df.columns:
#                     raise ValueError(f"Target column '{target_col}' not found in training data")
#                 if target_col not in test_df.columns:
#                     raise ValueError(f"Target column '{target_col}' not found in test data")
                
#                 # Check column consistency between train and test
#                 train_cols = set(train_df.columns)
#                 test_cols = set(test_df.columns)
                
#                 if train_cols != test_cols:
#                     missing_in_test = train_cols - test_cols
#                     missing_in_train = test_cols - train_cols
                    
#                     if missing_in_test:
#                         logger.error(f"Columns missing in test data: {missing_in_test}")
#                     if missing_in_train:
#                         logger.error(f"Columns missing in training data: {missing_in_train}")
                    
#                     raise ValueError("Training and test data have different columns")
                
#                 logger.info("Data consistency validation passed")
                
#             except Exception as e:
#                 logger.error(f"Data consistency validation failed: {str(e)}")
#                 raise CustomException("Data consistency validation failed", e)
            
#             # Split features and target
#             try:
#                 logger.info("Splitting features and target variables")
                
#                 # Training data split
#                 X_train = train_df.drop(columns=[target_col])
#                 y_train = train_df[target_col]
                
#                 # Test data split
#                 X_test = test_df.drop(columns=[target_col])
#                 y_test = test_df[target_col]
                
#                 # Validate splits
#                 if X_train.empty or y_train.empty:
#                     raise ValueError("Training features or target is empty after split")
#                 if X_test.empty or y_test.empty:
#                     raise ValueError("Test features or target is empty after split")
                
#                 # Check for missing values
#                 if X_train.isnull().any().any():
#                     null_cols = X_train.columns[X_train.isnull().any()].tolist()
#                     logger.warning(f"Training features contain null values in columns: {null_cols}")
                
#                 if X_test.isnull().any().any():
#                     null_cols = X_test.columns[X_test.isnull().any()].tolist()
#                     logger.warning(f"Test features contain null values in columns: {null_cols}")
                
#                 if y_train.isnull().any():
#                     logger.warning(f"Training target contains {y_train.isnull().sum()} null values")
                
#                 if y_test.isnull().any():
#                     logger.warning(f"Test target contains {y_test.isnull().sum()} null values")
                
#                 logger.info("Data splitting completed successfully")
#                 logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#                 logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                
#                 # Log class distribution
#                 train_class_dist = y_train.value_counts()
#                 test_class_dist = y_test.value_counts()
                
#                 logger.info("Training set class distribution:")
#                 for class_val, count in train_class_dist.items():
#                     percentage = (count / len(y_train)) * 100
#                     logger.info(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")
                
#                 logger.info("Test set class distribution:")
#                 for class_val, count in test_class_dist.items():
#                     percentage = (count / len(y_test)) * 100
#                     logger.info(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")
                
#                 logger.info("="*50)
#                 return X_train, y_train, X_test, y_test
                
#             except Exception as e:
#                 logger.error(f"Error during data splitting: {str(e)}")
#                 raise CustomException("Failed to split features and target", e)
                
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during data loading and splitting: {str(e)}")
#             raise CustomException("Data loading and splitting failed unexpectedly", e)
        
#     def train_lgbm(self, X_train, y_train):
#         """Train LightGBM model with hyperparameter tuning and comprehensive error handling"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING LIGHTGBM MODEL TRAINING")
#             logger.info("="*50)
            
#             # Validate input data
#             if X_train is None or X_train.empty:
#                 raise ValueError("Training features are None or empty")
#             if y_train is None or y_train.empty:
#                 raise ValueError("Training target is None or empty")
            
#             logger.info(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
            
#             # Check minimum data requirements
#             min_samples = max(self.random_search_params["cv"], 10)
#             if len(X_train) < min_samples:
#                 raise ValueError(f"Insufficient training samples: {len(X_train)} < {min_samples}")
            
#             # Check class distribution
#             unique_classes = y_train.nunique()
#             if unique_classes < 2:
#                 raise ValueError(f"Need at least 2 classes for classification, found: {unique_classes}")
            
#             # Initialize LightGBM model
#             try:
#                 logger.info("Initializing LightGBM classifier")
#                 lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])
#                 logger.info("LightGBM classifier initialized successfully")
                
#             except Exception as e:
#                 logger.error(f"Error initializing LightGBM model: {str(e)}")
#                 raise CustomException("Failed to initialize LightGBM classifier", e)
            
#             # Setup RandomizedSearchCV
#             try:
#                 logger.info("Setting up hyperparameter tuning with RandomizedSearchCV")
#                 logger.info(f"Parameter distributions: {self.params_dist}")
#                 logger.info(f"Random search configuration: {self.random_search_params}")
                
#                 random_search = RandomizedSearchCV(
#                     estimator=lgbm_model,
#                     param_distributions=self.params_dist,
#                     n_iter=self.random_search_params["n_iter"],
#                     cv=self.random_search_params["cv"],
#                     n_jobs=self.random_search_params["n_jobs"],
#                     verbose=self.random_search_params["verbose"],
#                     random_state=self.random_search_params["random_state"],
#                     scoring=self.random_search_params["scoring"]
#                 )
                
#                 logger.info("RandomizedSearchCV setup completed")
                
#             except Exception as e:
#                 logger.error(f"Error setting up RandomizedSearchCV: {str(e)}")
#                 raise CustomException("Failed to setup hyperparameter tuning", e)
            
#             # Perform hyperparameter tuning
#             try:
#                 logger.info("Starting hyperparameter tuning process")
#                 logger.info(f"This will perform {self.random_search_params['n_iter']} iterations with {self.random_search_params['cv']}-fold cross-validation")
                
#                 # Fit the model with hyperparameter tuning
#                 random_search.fit(X_train, y_train)
                
#                 logger.info("Hyperparameter tuning completed successfully")
                
#             except ValueError as ve:
#                 if "not enough" in str(ve).lower() or "cannot have" in str(ve).lower():
#                     logger.error(f"Cross-validation setup error: {str(ve)}")
#                     raise CustomException(f"Cross-validation failed - check data size and CV folds: {str(ve)}", ve)
#                 else:
#                     logger.error(f"Hyperparameter tuning validation error: {str(ve)}")
#                     raise CustomException(f"Hyperparameter tuning validation failed: {str(ve)}", ve)
#             except Exception as e:
#                 logger.error(f"Error during hyperparameter tuning: {str(e)}")
#                 raise CustomException("Hyperparameter tuning process failed", e)
            
#             # Extract best model and parameters
#             try:
#                 logger.info("Extracting best model and parameters")
                
#                 best_params = random_search.best_params_
#                 best_lgbm_model = random_search.best_estimator_
#                 best_score = random_search.best_score_
                
#                 # Validate results
#                 if best_params is None:
#                     raise ValueError("Best parameters are None")
#                 if best_lgbm_model is None:
#                     raise ValueError("Best model is None")
                
#                 logger.info("Hyperparameter tuning results:")
#                 logger.info(f"Best cross-validation score: {best_score:.4f}")
#                 logger.info(f"Best parameters: {best_params}")
                
#                 # Log parameter details
#                 for param_name, param_value in best_params.items():
#                     logger.info(f"  {param_name}: {param_value}")
                
#                 # Additional model information
#                 if hasattr(random_search, 'cv_results_'):
#                     logger.debug(f"Total parameter combinations tested: {len(random_search.cv_results_['params'])}")
                
#                 logger.info("="*50)
#                 return best_lgbm_model
                
#             except Exception as e:
#                 logger.error(f"Error extracting best model results: {str(e)}")
#                 raise CustomException("Failed to extract best model from hyperparameter tuning", e)
        
#         except ValueError as ve:
#             logger.error(f"Data validation error in model training: {str(ve)}")
#             raise CustomException(f"Model training validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during model training: {str(e)}")
#             raise CustomException("Model training failed unexpectedly", e)
    
#     def evaluate_model(self, model, X_test, y_test):
#         """Evaluate model performance with comprehensive error handling and validation"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING MODEL EVALUATION")
#             logger.info("="*50)
            
#             # Validate inputs
#             if model is None:
#                 raise ValueError("Model is None")
#             if X_test is None or X_test.empty:
#                 raise ValueError("Test features are None or empty")
#             if y_test is None or y_test.empty:
#                 raise ValueError("Test target is None or empty")
            
#             logger.info(f"Evaluation data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
            
#             # Check if model is fitted
#             try:
#                 # Try to access model attributes that should exist after fitting
#                 if not hasattr(model, 'classes_'):
#                     logger.warning("Model may not be fitted - 'classes_' attribute not found")
                
#             except Exception as e:
#                 logger.warning(f"Could not verify model fitting status: {str(e)}")
            
#             # Make predictions
#             try:
#                 logger.info("Making predictions on test data")
#                 y_pred = model.predict(X_test)
                
#                 # Validate predictions
#                 if y_pred is None:
#                     raise ValueError("Predictions are None")
#                 if len(y_pred) == 0:
#                     raise ValueError("Predictions array is empty")
#                 if len(y_pred) != len(y_test):
#                     raise ValueError(f"Prediction length mismatch: {len(y_pred)} != {len(y_test)}")
                
#                 logger.info(f"Predictions generated successfully - {len(y_pred)} predictions")
                
#                 # Log prediction distribution
#                 pred_dist = pd.Series(y_pred).value_counts()
#                 logger.info("Prediction distribution:")
#                 for class_val, count in pred_dist.items():
#                     percentage = (count / len(y_pred)) * 100
#                     logger.info(f"  Class {class_val}: {count} predictions ({percentage:.2f}%)")
                
#             except Exception as e:
#                 logger.error(f"Error making predictions: {str(e)}")
#                 raise CustomException("Failed to make predictions on test data", e)
            
#             # Calculate evaluation metrics
#             try:
#                 logger.info("Calculating evaluation metrics")
                
#                 # Calculate metrics with error handling
#                 try:
#                     accuracy = accuracy_score(y_test, y_pred)
#                     logger.debug(f"Accuracy calculated: {accuracy}")
#                 except Exception as e:
#                     logger.error(f"Error calculating accuracy: {str(e)}")
#                     raise CustomException("Failed to calculate accuracy score", e)
                
#                 try:
#                     precision = precision_score(y_test, y_pred, average='binary' if len(set(y_test)) == 2 else 'weighted', zero_division=0)
#                     logger.debug(f"Precision calculated: {precision}")
#                 except Exception as e:
#                     logger.error(f"Error calculating precision: {str(e)}")
#                     raise CustomException("Failed to calculate precision score", e)
                
#                 try:
#                     recall = recall_score(y_test, y_pred, average='binary' if len(set(y_test)) == 2 else 'weighted', zero_division=0)
#                     logger.debug(f"Recall calculated: {recall}")
#                 except Exception as e:
#                     logger.error(f"Error calculating recall: {str(e)}")
#                     raise CustomException("Failed to calculate recall score", e)
                
#                 try:
#                     f1 = f1_score(y_test, y_pred, average='binary' if len(set(y_test)) == 2 else 'weighted', zero_division=0)
#                     logger.debug(f"F1 score calculated: {f1}")
#                 except Exception as e:
#                     logger.error(f"Error calculating F1 score: {str(e)}")
#                     raise CustomException("Failed to calculate F1 score", e)
                
#                 # Log evaluation results
#                 logger.info("Model Evaluation Results:")
#                 logger.info(f"Accuracy Score: {accuracy:.4f}")
#                 logger.info(f"Precision Score: {precision:.4f}")
#                 logger.info(f"Recall Score: {recall:.4f}")
#                 logger.info(f"F1 Score: {f1:.4f}")
                
#                 # Create metrics dictionary
#                 metrics = {
#                     "accuracy": accuracy,
#                     "precision": precision,  # Fixed typo from "precison"
#                     "recall": recall,
#                     "f1": f1
#                 }
                
#                 # Validate metrics
#                 for metric_name, metric_value in metrics.items():
#                     if not isinstance(metric_value, (int, float)):
#                         raise ValueError(f"Invalid metric value for {metric_name}: {metric_value}")
#                     if not (0 <= metric_value <= 1):
#                         logger.warning(f"Metric {metric_name} outside expected range [0,1]: {metric_value}")
                
#                 logger.info("Model evaluation completed successfully")
#                 logger.info("="*50)
#                 return metrics
                
#             except CustomException:
#                 raise  # Re-raise custom exceptions
#             except Exception as e:
#                 logger.error(f"Error calculating evaluation metrics: {str(e)}")
#                 raise CustomException("Failed to calculate evaluation metrics", e)
        
#         except ValueError as ve:
#             logger.error(f"Data validation error in model evaluation: {str(ve)}")
#             raise CustomException(f"Model evaluation validation failed: {str(ve)}", ve)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during model evaluation: {str(e)}")
#             raise CustomException("Model evaluation failed unexpectedly", e)
        
#     def save_model(self, model):
#         """Save trained model with comprehensive error handling and validation"""
#         try:
#             logger.info("="*50)
#             logger.info("STARTING MODEL SAVING")
#             logger.info("="*50)
            
#             # Validate input model
#             if model is None:
#                 raise ValueError("Model is None")
            
#             logger.info(f"Saving model to: {self.model_output_path}")
            
#             # Validate and create directory
#             try:
#                 model_dir = os.path.dirname(self.model_output_path)
#                 if model_dir:
#                     if not os.path.exists(model_dir):
#                         logger.info(f"Creating model directory: {model_dir}")
#                         os.makedirs(model_dir, exist_ok=True)
                    
#                     # Check write permissions
#                     if not os.access(model_dir, os.W_OK):
#                         raise PermissionError(f"No write permission for directory: {model_dir}")
                
#             except OSError as e:
#                 logger.error(f"File system error creating model directory: {str(e)}")
#                 raise CustomException(f"Failed to create model directory: {model_dir}", e)
#             except PermissionError as e:
#                 logger.error(f"Permission error: {str(e)}")
#                 raise CustomException(f"Permission denied: {str(e)}", e)
            
#             # Check if file already exists and handle appropriately
#             if os.path.exists(self.model_output_path):
#                 logger.warning(f"Model file already exists and will be overwritten: {self.model_output_path}")
                
#                 # Check if existing file is writable
#                 if not os.access(self.model_output_path, os.W_OK):
#                     raise PermissionError(f"Existing model file is not writable: {self.model_output_path}")
            
#             # Save the model
#             try:
#                 logger.info("Serializing and saving the model using joblib")
                
#                 # Get model information before saving
#                 model_type = type(model).__name__
#                 logger.debug(f"Saving model of type: {model_type}")
                
#                 if hasattr(model, 'get_params'):
#                     logger.debug(f"Model has {len(model.get_params())} parameters")
                
#                 # Save model
#                 joblib.dump(model, self.model_output_path)
                
#                 # Verify model was saved
#                 if not os.path.exists(self.model_output_path):
#                     raise ValueError("Model file was not created")
                
#                 file_size = os.path.getsize(self.model_output_path)
#                 if file_size == 0:
#                     raise ValueError("Saved model file is empty")
                
#                 logger.info(f"Model saved successfully to: {self.model_output_path}")
#                 logger.info(f"Model file size: {file_size} bytes")
                
#                 # Test model loading to verify integrity
#                 try:
#                     logger.debug("Verifying model integrity by loading")
#                     test_model = joblib.load(self.model_output_path)
#                     if test_model is None:
#                         raise ValueError("Loaded model is None")
#                     logger.debug("Model integrity verification passed")
                    
#                 except Exception as e:
#                     logger.error(f"Model integrity verification failed: {str(e)}")
#                     # Clean up corrupted file
#                     try:
#                         os.remove(self.model_output_path)
#                         logger.debug("Removed corrupted model file")
#                     except:
#                         pass
#                     raise CustomException("Saved model failed integrity check", e)
                
#             except Exception as e:
#                 logger.error(f"Error saving model: {str(e)}")
#                 # Clean up partial file if it exists
#                 if os.path.exists(self.model_output_path):
#                     try:
#                         os.remove(self.model_output_path)
#                         logger.debug("Cleaned up partial model file")
#                     except:
#                         pass
#                 raise CustomException(f"Failed to save model to {self.model_output_path}", e)
            
#             logger.info("="*50)
            
#         except ValueError as ve:
#             logger.error(f"Model validation error: {str(ve)}")
#             raise CustomException(f"Model saving validation failed: {str(ve)}", ve)
#         except PermissionError as pe:
#             logger.error(f"Permission error during model saving: {str(pe)}")
#             raise CustomException(f"Permission denied for model saving: {str(pe)}", pe)
#         except CustomException:
#             raise  # Re-raise custom exceptions
#         except Exception as e:
#             logger.error(f"Unexpected error during model saving: {str(e)}")
#             raise CustomException("Model saving failed unexpectedly", e)
    
#     def run(self):
#         """Execute the complete model training pipeline with MLflow integration and comprehensive error handling"""
#         try:
#             logger.info("="*60)
#             logger.info("STARTING MODEL TRAINING PIPELINE")
#             logger.info("="*60)
            
#             # Start MLflow run with error handling
#             try:
#                 logger.info("Starting MLflow experiment run")
                
#                 with mlflow.start_run():
#                     logger.info("MLflow run started successfully")
                    
#                     # Get run information
#                     run = mlflow.active_run()
#                     if run:
#                         logger.info(f"MLflow run ID: {run.info.run_id}")
#                         logger.info(f"MLflow experiment ID: {run.info.experiment_id}")
                    
#                     # Step 1: Log datasets to MLflow
#                     logger.info("Step 1: Logging datasets to MLflow")
#                     try:
#                         logger.info("Logging training and test datasets to MLflow")
                        
#                         # Validate files exist before logging
#                         if not os.path.exists(self.train_path):
#                             raise FileNotFoundError(f"Training data file not found: {self.train_path}")
#                         if not os.path.exists(self.test_path):
#                             raise FileNotFoundError(f"Test data file not found: {self.test_path}")
                        
#                         mlflow.log_artifact(self.train_path, artifact_path="datasets")
#                         logger.debug(f"Training dataset logged: {self.train_path}")
                        
#                         mlflow.log_artifact(self.test_path, artifact_path="datasets")
#                         logger.debug(f"Test dataset logged: {self.test_path}")
                        
#                         logger.info("Datasets logged to MLflow successfully")
                        
#                     except Exception as e:
#                         logger.error(f"Error logging datasets to MLflow: {str(e)}")
#                         raise CustomException("Failed to log datasets to MLflow", e)
                    
#                     logger.info("Step 1 completed successfully")
                    
#                     # Step 2: Load and split data
#                     logger.info("Step 2: Loading and splitting data")
#                     try:
#                         X_train, y_train, X_test, y_test = self.load_and_split_data()
#                         logger.info("Data loading and splitting completed successfully")
                        
#                     except Exception as e:
#                         logger.error(f"Error in data loading and splitting: {str(e)}")
#                         raise CustomException("Data loading and splitting step failed", e)
                    
#                     logger.info("Step 2 completed successfully")
                    
#                     # Step 3: Train model
#                     logger.info("Step 3: Training LightGBM model")
#                     try:
#                         best_lgbm_model = self.train_lgbm(X_train, y_train)
#                         logger.info("Model training completed successfully")
                        
#                     except Exception as e:
#                         logger.error(f"Error in model training: {str(e)}")
#                         raise CustomException("Model training step failed", e)
                    
#                     logger.info("Step 3 completed successfully")
                    
#                     # Step 4: Evaluate model
#                     logger.info("Step 4: Evaluating model performance")
#                     try:
#                         metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
#                         logger.info("Model evaluation completed successfully")
                        
#                     except Exception as e:
#                         logger.error(f"Error in model evaluation: {str(e)}")
#                         raise CustomException("Model evaluation step failed", e)
                    
#                     logger.info("Step 4 completed successfully")
                    
#                     # Step 5: Save model
#                     logger.info("Step 5: Saving trained model")
#                     try:
#                         self.save_model(best_lgbm_model)
#                         logger.info("Model saving completed successfully")
                        
#                     except Exception as e:
#                         logger.error(f"Error in model saving: {str(e)}")
#                         raise CustomException("Model saving step failed", e)
                    
#                     logger.info("Step 5 completed successfully")
                    
#                     # Step 6: Log model and metadata to MLflow
#                     logger.info("Step 6: Logging model and metadata to MLflow")
#                     try:
#                         # Log model artifact
#                         logger.info("Logging model artifact to MLflow")
#                         if not os.path.exists(self.model_output_path):
#                             raise FileNotFoundError(f"Model file not found for logging: {self.model_output_path}")
                        
#                         mlflow.log_artifact(self.model_output_path)
#                         logger.debug(f"Model artifact logged: {self.model_output_path}")
                        
#                         # Log model parameters
#                         logger.info("Logging model parameters to MLflow")
#                         try:
#                             model_params = best_lgbm_model.get_params()
#                             if not model_params:
#                                 logger.warning("No model parameters found to log")
#                             else:
#                                 # Filter out None values and convert complex types to strings
#                                 filtered_params = {}
#                                 for key, value in model_params.items():
#                                     if value is not None:
#                                         # Convert complex types to strings for MLflow compatibility
#                                         if isinstance(value, (list, dict, tuple)):
#                                             filtered_params[key] = str(value)
#                                         else:
#                                             filtered_params[key] = value
                                
#                                 mlflow.log_params(filtered_params)
#                                 logger.debug(f"Logged {len(filtered_params)} model parameters")
                            
#                         except Exception as e:
#                             logger.error(f"Error logging model parameters: {str(e)}")
#                             raise CustomException("Failed to log model parameters to MLflow", e)
                        
#                         # Log evaluation metrics
#                         logger.info("Logging evaluation metrics to MLflow")
#                         try:
#                             if not metrics:
#                                 logger.warning("No metrics found to log")
#                             else:
#                                 # Validate metrics before logging
#                                 validated_metrics = {}
#                                 for metric_name, metric_value in metrics.items():
#                                     if isinstance(metric_value, (int, float)) and not (metric_value != metric_value):  # Check for NaN
#                                         validated_metrics[metric_name] = float(metric_value)
#                                     else:
#                                         logger.warning(f"Skipping invalid metric {metric_name}: {metric_value}")
                                
#                                 if validated_metrics:
#                                     mlflow.log_metrics(validated_metrics)
#                                     logger.debug(f"Logged {len(validated_metrics)} evaluation metrics")
#                                 else:
#                                     logger.warning("No valid metrics to log")
                            
#                         except Exception as e:
#                             logger.error(f"Error logging evaluation metrics: {str(e)}")
#                             raise CustomException("Failed to log evaluation metrics to MLflow", e)
                        
#                         # Log additional metadata
#                         try:
#                             logger.info("Logging additional metadata to MLflow")
                            
#                             # Log data shapes
#                             mlflow.log_param("train_samples", len(X_train))
#                             mlflow.log_param("test_samples", len(X_test))
#                             mlflow.log_param("n_features", X_train.shape[1])
#                             mlflow.log_param("n_classes", len(set(y_train)))
                            
#                             # Log file paths (for reference)
#                             mlflow.log_param("train_data_path", os.path.basename(self.train_path))
#                             mlflow.log_param("test_data_path", os.path.basename(self.test_path))
#                             mlflow.log_param("model_output_path", os.path.basename(self.model_output_path))
                            
#                             logger.debug("Additional metadata logged successfully")
                            
#                         except Exception as e:
#                             logger.warning(f"Error logging additional metadata: {str(e)}")
#                             # Don't fail the pipeline for metadata logging errors
                        
#                         logger.info("MLflow logging completed successfully")
                        
#                     except CustomException:
#                         raise  # Re-raise custom exceptions
#                     except Exception as e:
#                         logger.error(f"Error in MLflow logging: {str(e)}")
#                         raise CustomException("MLflow logging step failed", e)
                    
#                     logger.info("Step 6 completed successfully")
                    
#                     # Pipeline completion summary
#                     logger.info("="*60)
#                     logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
#                     logger.info("="*60)
#                     logger.info(f"Training samples: {len(X_train)}")
#                     logger.info(f"Test samples: {len(X_test)}")
#                     logger.info(f"Features: {X_train.shape[1]}")
#                     logger.info(f"Model saved to: {self.model_output_path}")
#                     logger.info("Final model performance:")
#                     for metric_name, metric_value in metrics.items():
#                         logger.info(f"  {metric_name.capitalize()}: {metric_value:.4f}")
#                     logger.info("="*60)
            
#             except Exception as e:
#                 logger.error(f"Error in MLflow run context: {str(e)}")
#                 raise CustomException("MLflow run execution failed", e)
                
#         except CustomException as ce:
#             logger.error("="*60)
#             logger.error(f"MODEL TRAINING PIPELINE FAILED - CustomException: {str(ce)}")
#             logger.error("="*60)
#             raise ce
        
#         except Exception as e:
#             logger.error("="*60)
#             logger.error(f"MODEL TRAINING PIPELINE FAILED - Unexpected error: {str(e)}")
#             logger.error("="*60)
#             raise CustomException("Model training pipeline failed unexpectedly", e)
        
#         finally:
#             logger.info("Model training pipeline execution finished")

# if __name__ == "__main__":
#     try:
#         logger.info("Starting model training script")
        
#         # Validate required paths exist
#         required_paths = [PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH]
#         path_names = ["PROCESSED_TRAIN_DATA_PATH", "PROCESSED_TEST_DATA_PATH"]
        
#         for path, name in zip(required_paths, path_names):
#             if not os.path.exists(path):
#                 logger.error(f"Required path does not exist - {name}: {path}")
#                 raise FileNotFoundError(f"Required file not found - {name}: {path}")
        
#         # Validate model output directory can be created
#         model_dir = os.path.dirname(MODEL_OUTPUT_PATH)
#         if model_dir and not os.path.exists(model_dir):
#             try:
#                 os.makedirs(model_dir, exist_ok=True)
#                 logger.info(f"Created model output directory: {model_dir}")
#             except OSError as e:
#                 logger.error(f"Cannot create model output directory {model_dir}: {str(e)}")
#                 raise CustomException(f"Failed to create model output directory: {model_dir}", e)
        
#         logger.info("All required paths validated successfully")
        
#         # Check MLflow availability
#         try:
#             import mlflow
#             logger.info(f"MLflow version: {mlflow.__version__}")
#         except ImportError as e:
#             logger.error("MLflow is not available")
#             raise CustomException("MLflow is required but not installed", e)
        
#         # Initialize and run model trainer
#         trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
#         trainer.run()
        
#         logger.info("Model training script completed successfully")
        
#     except CustomException as ce:
#         logger.error(f"Script failed with CustomException: {str(ce)}")
#         exit(1)
    
#     except FileNotFoundError as fe:
#         logger.error(f"Script failed - Required file not found: {str(fe)}")
#         exit(1)
    
#     except ImportError as ie:
#         logger.error(f"Script failed - Import error: {str(ie)}")
#         exit(1)
    
#     except Exception as e:
#         logger.error(f"Script failed with unexpected error: {str(e)}")
#         raise CustomException("Model training script failed", e)    

            