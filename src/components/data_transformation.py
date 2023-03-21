import os
import sys 
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This fuction is responsible for data transformation.
        """
        try:
            categorical_features = ['Month', 'VisitorType','OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']
            numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

            # num_pipeline=Pipeline(
            #     steps=[
            #     ("imputer",SimpleImputer(strategy="median")),
            #     ("scaler",StandardScaler())
            # ])

            
            # cat_pipeline=Pipeline(
            #     steps=[
            #     ("imputer",SimpleImputer()),
            #     ("one_hot_encoder",OneHotEncoder),
            #     ("saceler",StandardScaler())
            # ])
            num_pipeline=Pipeline(
                steps=[
                ("scaler",StandardScaler())
            ])
            
            logging.info(f"Numerical Column Column : {numerical_features}")
            
            cat_pipeline=Pipeline(
                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Column : {categorical_features}")


            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
                ])
            logging.info(f"PRE PRO Column : {preprocessor}")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
        

    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path) 
                
                logging.info("Read Train and Test Data")

                logging.info("Obtaining Preprocessing Object")
                
                preprocessing_obj = self.get_data_transformer_object()

                target_column_name="Revenue"
                numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]

                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]

                logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
                print(input_feature_train_arr.shape,input_feature_test_arr.shape)
                print(target_feature_train_df.shape,target_feature_test_df.shape)
                print("*"*44)
                input_feature_train_arr = pd.DataFrame(input_feature_train_arr.toarray())
                input_feature_test_arr = pd.DataFrame(input_feature_test_arr.toarray())
                print(type(input_feature_train_arr))
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                
                #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


                logging.info(f"Saved preprocessing object.")

                save_object(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj

                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
            except Exception as e:
                raise CustomException(e,sys) # type: ignore
