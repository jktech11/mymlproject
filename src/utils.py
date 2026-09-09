import os
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import numpy as np 
import pandas as pd
import pickle
import dill

from src.exception import CustomException

def save_object(obj,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,para):
    try:
        
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = para[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model,cv=3,param_grid=params)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            predict = model.predict(X_test)
            score = r2_score(y_test,predict)

            report[list(models.keys())[i]] = score

            return report

    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_object:
            obj = pickle.load(file_object)
        return obj
    
    except Exception as e:
        raise CustomException(e,sys)

    