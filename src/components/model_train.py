import os 
import sys
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegression

from dataclasses import dataclass
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_model_trianer(self,train_arr,test_arr):
        try:
            logging.info("split train and test data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("tpot library is being implemented...")
            

            # Initialize TPOTClassifier with the desired configuration

            # Initialize TPOTClassifier with the desired configuration
            tpot = TPOTClassifier(
                generations=5,
                population_size=20,
                verbosity=2,
                scoring='roc_auc',
                random_state=42,
                disable_update_check=True,
                config_dict='TPOT light'
            )

            # Fit the TPOT model on the training data
            tpot.fit(X_train, y_train)

            # Assign the best pipeline (model) found by TPOT to a variable
            best_model = tpot.fitted_pipeline_


            # Calculate the AUC score for the TPOT model
            tpot_auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            print(f"\nAUC score: {tpot_auc_score:.4f}")

            # Print the best pipeline steps
            print('\nBest pipeline steps:', end='\n')
            for idx, (name, transform) in enumerate(best_model.steps, start=1):
                print(f'{idx}. {transform}')

            logreg = LogisticRegression(random_state=42)
            logreg.fit(X_train, y_train)


            # Make predictions on the test set
            y_pred_proba = logreg.predict_proba(X_test)[:, 1]
            # Calculate the AUC score
            logreg_auc_score = roc_auc_score(y_test, y_pred_proba)

            if logreg_auc_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            return(logreg_auc_score)

        except Exception as e:
            raise CustomException(e,sys)