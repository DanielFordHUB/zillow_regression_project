import pandas as pd
import os
from env import get_db_url
import numpy as np
import math
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer



def get_zillow_data():
    """Seeks to read the cached zillow.csv first """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()



def get_new_zillow_data():
    '''this function gathers selected data from the ZILLOW SQL DF
    and uses the get_db_url function to connect to said dataframe'''
    sql = '''
    SELECT 
        bedroomcnt AS bedrooms, 
        bathroomcnt AS bathrooms,
        calculatedfinishedsquarefeet AS sq_ft,
        taxvaluedollarcnt AS tax_value,
        yearbuilt AS year_built,
        lotsizesquarefeet AS lot_size,
        fips
    FROM
        properties_2017
       JOIN propertylandusetype using (propertylandusetypeid)
       JOIN predictions_2017 USING(parcelid)
    WHERE propertylandusedesc in ("Single Family Residential", 
                                  "Inferred Single Family Residential")
       AND transactiondate LIKE "2017%%";          
    '''
    return pd.read_sql(sql, get_db_url('zillow'))


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    #drop the 121 duplicate values
    df = df.drop_duplicates()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # yearbuilt, and bedrooms can be integers
    df["year_built"] = df["year_built"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)    
    df["tax_value"] = df["tax_value"].astype(int)
    df["sq_ft"] = df["sq_ft"].astype(int)
    df['lot_size'] = df['lot_size'].astype(int)
    
    #df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    # Turn fips to obj
    df['fips'] = df['fips'].astype(str)
    # Encode fips
    dummy_df = pd.get_dummies(df[['fips']], dummy_na=False)

    df = pd.concat([df, dummy_df], axis=1)
    
    #drop fips
    df = df.drop(['fips'], axis = 1)
    # rename fips
    df = df.rename(columns = {'fips_6037.0': 'LA', 'fips_6059.0':'orange', 'fips_6111.0': 'ventura'})
    
    return df


def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def remove_outliers(df):
    #create new column in dataframe that contains Mahalanobis distance for each row
    df['mahalanobis'] = mahalanobis(x=df, data=df[['bedrooms',
 'bathrooms',
 'sq_ft',
 'tax_value',
 'year_built',
 'lot_size',
 'fips']])
    
    #calculate p-value for each mahalanobis distance 
    df['p'] = 1 - chi2.cdf(df['mahalanobis'], 3)
    
    # drop rowss with p-value of less than 0.001
    df = df[df.p > 0.001]
    
    # drop calculative columns
    df = df.drop(['mahalanobis', 'p'], axis = 1)
    
    return df


def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    df = handle_nulls(df)

    df = remove_outliers(df)

    df = optimize_types(df)

    df.to_csv("zillow.csv", index=False)

    return df




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


## TODO Encode categorical variables (and FIPS is a category so Fips to string to one-hot-encoding
## TODO Scale numeric columns
## TODO Add train/validate/test split in here
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls
