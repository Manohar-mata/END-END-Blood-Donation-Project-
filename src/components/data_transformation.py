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
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformer_object(self):
        logging.info("The data transformation is started")
        try:
            columns = ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months']
            #data=data.drop_duplicates(inplace=True)
            data_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("Scale",StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer(
                [
                    ("datascale",data_pipeline,columns)
                ]
            )

            logging.info("The data processing is done!!!")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            train_df.columns= ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months', 'donated_march_2007']
            test_df.columns = ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months', 'donated_march_2007']

            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            logging.info("data duplicates are removed")


            logging.info("Reading train and test data completed")
            logging.info("obtaining the preprocessor object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column='donated_march_2007'

            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test=test_df[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            #input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info(f"Saved preprocessing object.")

            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)



            






        

