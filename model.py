import pandas as pd
import os
from env import get_db_url
import numpy as np
import math
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer



def train_test_validate_split(df, test_size=.2, validate_size=.3, random_state=99):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=99)
    train, validate = train_test_split(train, test_size=.3, random_state=99)
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate


def MM_scale_zillow(train, validate, test):
    '''
    This takes in the train, validate, and test dataframes, as well as the target label. 
    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the train, validate, and test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 
    train, validate, and test dataframes are returned, in that order. 
    '''
    target = 'taxvalue'
    
    # identify quantitative features to scale
    quant_features = [col for col in train.columns if (train[col].dtype != 'object') & (col != target)]
    
    # establish empty dataframes for storing scaled dataset
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # create and fit the scaler
    scaler = MinMaxScaler().fit(train[quant_features])
    
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
   

    return train_scaled, validate_scaled, test_scaled