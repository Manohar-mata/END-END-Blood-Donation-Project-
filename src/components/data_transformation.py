import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        """Creates a preprocessing pipeline object for data transformation."""
        logging.info("Starting data transformation pipeline creation.")
        try:
            columns = ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months']

            # Preprocessing pipeline for numeric data
            data_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            # ColumnTransformer to apply pipeline to specified columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("data_scaler", data_pipeline, columns)
                ]
            )

            logging.info("Data preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error occurred during pipeline creation.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Performs data transformation on train and test data."""
        try:
            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path)

            logging.info(f"Reading testing data from: {test_path}")
            test_df = pd.read_csv(test_path)

            # Set expected column names
            expected_columns = ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months', 'donated_march_2007']
            train_df.columns = expected_columns
            test_df.columns = expected_columns
            #print(train_df.columns,test_df.columns)

            # Remove duplicate entries
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            logging.info("Duplicates removed from training and testing datasets.")

            logging.info("Obtaining the preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'donated_march_2007'

            # Separate input features and target variable for training and testing
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("Preprocessing complete. Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessing object saved successfully.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(f"Ensure the file paths are correct. Error: {e}", sys)
        except KeyError as e:
            logging.error(f"Missing columns in input data: {e}")
            raise CustomException(f"Ensure the dataset contains the required columns: {e}", sys)
        except Exception as e:
            logging.error("An error occurred during data transformation.")
            raise CustomException(e, sys)
