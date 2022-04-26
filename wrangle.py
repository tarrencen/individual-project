import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import get_db_url
from acquire import zachs_zillow_pull
from pydataset import data
from sklearn.model_selection import train_test_split
import os

def show_codeup_dbs():
    '''
    Returns a list of the databases residing the Codeup SQL server
    '''
    url = get_db_url('employees')
    codeup_dbs = pd.read_sql('SHOW DATABASES', url)
    print('List of Codeup DBs:\n')
    return codeup_dbs

def get_prop_vals():
    '''
    Returns a DataFrame composed of selected column data from the properties_2017 table in the zillow database on
    Codeup's SQL serve
    '''
    filename = 'prop_vals.csv'
    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)
    query = ''' 
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid
    FROM properties_2017
    '''
    print('Getting a fresh copy from SQL database...')
    url = get_db_url('zillow')
    prop_vals = pd.read_sql(query, url)
    print('Copying to CSV...')
    prop_vals.to_csv(filename)
    return prop_vals

def wrangle_zillow():
    '''
    Returns a cleaned subset of prop_vals DataFrame
    '''
    prop_vals = get_prop_vals()
    prop_vals = prop_vals.rename(columns={
    'bedroomcnt': 'beds', 
    'bathroomcnt': 'baths', 
    'roomcnt': 'total_rooms',
    'numberofstories': 'stories',
    'fireplaceflag': 'fireplace',
    'poolcnt': 'pools', 
    'buildingqualitytypeid': 'condition', 
    'calculatedfinishedsquarefeet': 'calc_fin_sqft', 
    'lotsizesquarefeet': 'lot_sqft', 
    'structuretaxvaluedollarcnt': 'structure_tax_val',
    'landtaxvaluedollarcnt': 'land_tax_val',
    'taxvaluedollarcnt': 'tax_val', 
    'yearbuilt': 'yr_built', 
    'taxamount': 'tax_amt', 
    'fips': 'county_code', 
    'logerror': 'log_err' 
    })
    prop_vals_clean = prop_vals.dropna()
    prop_vals_clean.bedrooms = prop_vals_clean.bedrooms.astype('int')
    prop_vals_clean.bathrooms = prop_vals_clean.bathrooms.astype('int')
    prop_vals_clean.calcfin_sqft = prop_vals_clean.calcfin_sqft.astype('int')
    prop_vals_clean.tax_val = prop_vals_clean.tax_val.astype('int')
    prop_vals_clean.yr_built = prop_vals_clean.yr_built.astype('int')
    prop_vals_clean.fips = prop_vals_clean.fips.astype('int')
    train, test = train_test_split(prop_vals_clean, train_size=0.8, random_state=302)
    train, validate = train_test_split(train, test_size=0.7, random_state=302)
    return prop_vals_clean, train, validate, test

def wrangle_zillow_cluster(df):
    '''
    Takes in a dataframe of raw Zillow data acquired by a SQL query of the Codeup MySQL database and
    returns a clean version of the dataframe.'''

    df = zachs_zillow_pull()
    # Drop columns that contain >= 67% nulls AND not suitable for imputation
    df = df.drop(columns= [
    'id',
    'airconditioningtypeid',
    'architecturalstyletypeid',
    'basementsqft',
    'buildingclasstypeid',
    'decktypeid',
    'finishedfloor1squarefeet',
    'finishedsquarefeet13',
    'finishedsquarefeet15',
    'finishedsquarefeet50',
    'finishedsquarefeet6',
    'fireplacecnt',
    'garagecarcnt',
    'garagetotalsqft',
    'hashottuborspa',
    'poolsizesum',
    'pooltypeid10',
    'pooltypeid2',
    'pooltypeid7',
    'storytypeid',
    'threequarterbathnbr',
    'typeconstructiontypeid',
    'yardbuildingsqft17',
    'yardbuildingsqft26',
    'numberofstories', 
    'taxdelinquencyflag',
    'taxdelinquencyyear',
    'airconditioningdesc',
    'architecturalstyledesc',
    'buildingclassdesc',
    'storydesc',
    'typeconstructiondesc'])
    
    # Impute values as appropriate for feature; consult data dictionary for best imputation option
    df.buildingqualitytypeid = df.buildingqualitytypeid.fillna(df.buildingqualitytypeid.median())
    df.fireplaceflag = df.fireplaceflag.fillna(0)
    df.poolcnt = df.poolcnt.fillna(0)
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('Yes')
    df.heatingorsystemtypeid = df.heatingorsystemtypeid.fillna(24)
    df.fullbathcnt = df.fullbathcnt.fillna(2)
    df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(round(df.finishedsquarefeet12.mean()))
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(round(df.calculatedfinishedsquarefeet.mean()))
    df.unitcnt = df.unitcnt.fillna(1)
    df.yearbuilt = df.yearbuilt.fillna(df.yearbuilt.median())
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(round(df.taxvaluedollarcnt.mean(),2))
    df.taxamount = df.taxamount.fillna(round(df.taxamount.mean(),2))
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(2)
    
    # Make column names more Python-digestible
    df = df.rename(columns= {
        'parcelid': 'parcel_id', 
        'bathroomcnt': 'baths', 
        'bedroomcnt': 'beds',
        'buildingqualitytypeid': 'bldg_qual',
        'calculatedbathnbr': 'calc_bath', 
        'calculatedfinishedsquarefeet': 'calc_fin_sqft',
        'finishedsquarefeet12': 'fin_sqft_12', 
        'fips': 'county_name', 
        'fullbathcnt': 'full_bath_ct', 
        'heatingorsystemtypeid': 'heat_sys',
        'latitude': 'lat', 
        'longitude': 'long', 
        'poolcnt': 'has_pool', 
        'propertycountylandusecode': 'county_prop_code',
        'propertylandusetypeid': 'prop_land_id', 
        'rawcensustractandblock': 'raw_cens_block', 
        'regionidzip': 'zip',
        'roomcnt': 'room_ct', 
        'unitcnt': 'unit_ct', 
        'yearbuilt': 'yr_blt', 
        'fireplaceflag': 'has_firepl',
        'structuretaxvaluedollarcnt': 'structure_tax_val', 
        'taxvaluedollarcnt': 'tax_val', 
        'assessmentyear': 'yr_assess',
        'landtaxvaluedollarcnt': 'land_tax_val', 
        'taxamount': 'tax_amt', 
        'logerror': 'log_err', 
        'transactiondate': 'transact_date',
        'heatingorsystemdesc': 'heat_sys_desc', 
        'propertylandusedesc': 'prop_land_desc'
         })

    # Add 'prop_age' column
    df['prop_age'] = 2017 - df.yr_blt


    # Drop columns with redundant or irrelevant information
    df = df.drop(columns= [
        'lotsizesquarefeet',
        'propertyzoningdesc',
        'regionidcounty',
        'regionidcity', 
        'regionidneighborhood',
        'censustractandblock', 
        'raw_cens_block'
        ])
    # Drop remaining nulls
    clean_df = df.dropna()

    # Reduce dataset to Single Family Residences only and drop columns accordingly
    clean_df = clean_df[clean_df.prop_land_id == 261]
    clean_df = clean_df.drop(columns= ['prop_land_id'])


    # Resolve remaining data type issues and drop any other unnecessary columns
    clean_df.beds = clean_df.beds.astype('int64')
    clean_df.bldg_qual = clean_df.bldg_qual.astype('int64')
    clean_df.county_name = clean_df.county_name.replace({6037: 'LA County', 6059: 'Orange County', 6111: 'Ventura County'})
    clean_df.full_bath_ct = clean_df.full_bath_ct.astype('int64')
    clean_df.heat_sys = clean_df.heat_sys.astype('int64')
    clean_df.has_pool = clean_df.has_pool.astype('uint8')
    clean_df.zip = clean_df.zip.astype('int64')
    clean_df.room_ct = clean_df.room_ct.astype('int64')
    clean_df.unit_ct = clean_df.unit_ct.astype('int64')
    clean_df.has_firepl = clean_df.has_firepl.astype('uint8')
    clean_df = clean_df.drop(columns= ['yr_blt', 'structure_tax_val', 'yr_assess', 'land_tax_val', 'tax_amt'])
    clean_df.prop_age = clean_df.prop_age.astype('int64')


    return clean_df





def col_null_analysis(df):
    '''
    Takes in a df and returns a new dataframe containing the original df's null counts and percentages
    '''
    null_counts = df.isna().sum()
    null_pct = df.isna().mean() * 100
    df_nulls = pd.DataFrame({'null_counts': null_counts, 'null_pct': null_pct})
    return df_nulls

def row_null_analysis(df):
    '''
    Takes in a df and returns a new dataframe with the original df's null counts and percentages
    by row
    '''
    row_null_count = df.isna().sum(axis=1)
    row_null_pct = df.isna().mean(axis=1) * 100
    df_null_rows = pd.DataFrame({'row_null_count': row_null_count, 'row_null_pct': row_null_pct})
    return df_null_rows
