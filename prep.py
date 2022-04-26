import pandas as pd
import numpy as np
import wrangle as wr
from env import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

def get_splits(df):
    train, test = train_test_split(df, test_size= 0.2, random_state=302)
    train, validate = train_test_split(train, test_size= 0.3, random_state=302)
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
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_Xy_dummies(X_train, X_validate, X_test):
    '''
    Takes in X train/validate/test splits (target variable already removed) and returns reassigned splits with one-hot encoded categorical variables ("dummies")
    concatenated as part of the new dataframe
    '''
    X_train_dummies = pd.get_dummies(X_train.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_train = pd.concat([X_train, X_train_dummies], axis=1, ignore_index=False)

    X_validate_dummies = pd.get_dummies(X_validate.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_validate = pd.concat([X_validate, X_validate_dummies], axis=1, ignore_index=False)

    X_test_dummies = pd.get_dummies(X_test.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_test = pd.concat([X_test, X_test_dummies], axis=1, ignore_index=False)
    return X_train, X_validate, X_test

def minmax_scale_data(train, validate, test, cols_to_scale, return_scaler=False):
    '''
    Takes in train/validate/test splits of a prepared dataset, a list of its columns to scale, 
    and an option to return the scaler; returns scaled versions of the splits and the scaler
    if the option is set to True.'''

    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    mmscaler = MinMaxScaler()

    mmscaler.fit(train[cols_to_scale])

    train_scaled[cols_to_scale] = mmscaler.transform(train[cols_to_scale])
    validate_scaled[cols_to_scale] = mmscaler.transform(validate[cols_to_scale])
    test_scaled[cols_to_scale] = mmscaler.transform(test[cols_to_scale])

    if return_scaler:
        return mmscaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


def rfe(X, y, k):
    lm = LinearRegression()
    rfe = RFE(lm, k)
    rfe.fit(X, y)

    rfe_mask = rfe.support_
    rfe_feature = X.iloc[:,rfe_mask].columns.tolist()
    var_ranks = rfe.ranking_
    var_names = X.columns.tolist()
    rfe_ranked = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    return rfe_feature, rfe_ranked

def select_kbest(X, y, k):
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X,y)
    f_mask = f_selector.get_support()
    f_feature = X.iloc[:,f_mask].columns.tolist()
    return f_feature

def pipe_fitter(X_scaled, y):
    '''
    Takes in a scaled, isolated train/validate/test split and a target df, fits them to an OLS model, and 
    returns the target df with a column of predictions(['yhat_']) from the model.'''
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    y['yhat_'] = lr.predict(X_scaled)
    return y