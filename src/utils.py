import sys
import os
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
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


import dill
import pandas as pd
import numpy as np

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report ={}
       

        for i in range(len(list(models))):
            model =list(models.values())[i]
            if model == KNeighborsClassifier() or model == DecisionTreeClassifier():
                X_train = X_train.toarray()
            else:
                pass
                
            
            model.fit(X_train,y_train) # Train model


            #Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Evaluate Train and Val dataset
            model_train_roc_auc =roc_auc_score(y_train, y_train_pred, average='macro', multi_class='ovo')
            model_test_roc_auc =roc_auc_score(y_test, y_test_pred, average='macro', multi_class='ovo')

    

            report[list(models.keys())[i]] = model_test_roc_auc

        
        

            

        return report


    except Exception as e:
        raise CustomException(e,sys)
