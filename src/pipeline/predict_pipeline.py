import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object  # Assuming you have a utility function for loading objects

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\\model.pkl'
            preprocessor_path = 'artifacts\\preprocessor.pkl'
            
            print("Before Loading Model and Preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading Model and Preprocessor")
            
            # Check if the necessary columns are in the data
            required_columns = ['recency_months', 'frequency_times', 'monetary_cc_blood', 'time_months']
            missing_columns = [col for col in required_columns if col not in features.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")
            
            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Predict using the model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Recency_months: int, Frequency_times: int, Monetary_cc_blood: int, Time_months: int):
        self.Recency_months = Recency_months
        self.Frequency_times = Frequency_times
        self.Monetary_cc_blood = Monetary_cc_blood
        self.Time_months = Time_months

    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                "recency_months": [self.Recency_months],
                "frequency_times": [self.Frequency_times],
                "monetary_cc_blood": [self.Monetary_cc_blood],
                "time_months": [self.Time_months]
            }

            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)


