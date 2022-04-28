import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import env
import acquire as acq
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

def get_splits(df):
    train_validate, test = train_test_split(df, train_size= 0.8, random_state=302)
    train, validate = train_test_split(train_validate, train_size= 0.7, random_state=302)
    return train, validate, test
    


def isolate_lm_target(train, validate, test, target):
    '''
    Takes in train/validate/test splits and a target variable and returns corresponding X and y splits with
    target variable isolated (y_train, y_validate, y_test), ready for modeling.
    '''
    X_train = train.drop(columns= [target])
    y_train = train[[target]]

    X_validate = validate.drop(columns= [target])
    y_validate = validate[[target]]

    X_test = test.drop(columns= [target])
    y_test= test[[target]]

    X_train_dummies = pd.get_dummies(X_train.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_train = pd.concat([X_train, X_train_dummies], axis=1, ignore_index=False)

    X_validate_dummies = pd.get_dummies(X_validate.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_validate = pd.concat([X_validate, X_validate_dummies], axis=1, ignore_index=False)

    X_test_dummies = pd.get_dummies(X_test.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_test = pd.concat([X_test, X_test_dummies], axis=1, ignore_index=False)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_rmse(y_target, y_pred):
    '''
    Takes in a target variable and a regression model's predictions on that variable and
    return the root mean squared error for the model's predictions.'''
    rmse = mean_squared_error(y_target, y_pred) ** 0.5
    return rmse


    




