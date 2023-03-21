import pandas as pd
import numpy as np

import sys
import os

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

#Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

#hyperparameter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    #def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],)

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Machine": SVC(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "XGBoost Classifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "CatBoost Classifier": CatBoostClassifier()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test =X_test,y_test=y_test,models=models)

            # to get best model
            best_model_score = max(sorted(model_report.values()))

            # to get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException ("No best model not found",sys) # type: ignore
            
            logging.info("best model found")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            

            predicted = best_model.predict(X_test)

            roc_auc_score_ = roc_auc_score(y_test,predicted)
            logging.info(f"models: {model_report}")   
            return roc_auc_score_

        except Exception as e:
            raise CustomException(e,sys) # type: ignore
