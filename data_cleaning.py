import pandas as pd
import numpy as np

def remove_missing_value(df):

   #check amount of missing value in each colunm and decide if missing value is dropped off or filled in aggregated values
    for colunm_name in df.columns:
        total_missing_count = df[colunm_name].isnull().sum()

        if total_missing_count != 0:
            print(f'{colunm_name} has {total_missing_count} missing values')

    #here we just chose to drop off missing values 
    #in other scenarios we can also choose to fill in aggregated values such as mean, median or mode
    df.dropna(axis = 0, inplace = True)

def deduplicate(df):
    for colunm_name in df.columns:
        if df.duplicated(subset = [colunm_name], keep = 'first') == 'False':
            df.drop_duplicates()

def drop_outlier_data(df, ignored_column_list):
    #Filter out data outliers
    #use IQR (Inter Quartile Range)-IQR = Quartile3 – Quartile1
    for colunm_name in df.columns:
        if df[colunm_name].dtypes == 'object' or colunm_name in ignored_column_list:
            continue
        Q1 = np.percentile(df[colunm_name], 5, method = 'midpoint')
        Q3 = np.percentile(df[colunm_name], 95, method = 'midpoint')

        IQR = Q3 - Q1      

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        df_upper_bound = df.loc[df[colunm_name] > upper_bound]
        df_lower_bound = df.loc[df[colunm_name] < lower_bound]

        df.drop(df_upper_bound, inplace = True, errors = 'ignore')
        df.drop(df_lower_bound, inplace = True, errors = 'ignore')

def clean_price_demand_dataset(df_price_demand):
    
    df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'], format='%d/%m/%Y %H:%M', errors='coerce')

    remove_missing_value(df_price_demand)

    deduplicate(df_price_demand)

    drop_outlier_data(df_price_demand, ['SETTLEMENTDATE'])

    return df_price_demand
  
def clean_weather_dataset(df_weather):

    #convert data types to datetime
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d/%m/%Y', errors='coerce')

    #convert object to float 
    df_weather['9am wind speed (km/h)'] = pd.to_numeric(df_weather['9am wind speed (km/h)'], errors='coerce')
    df_weather['3pm wind speed (km/h)'] = pd.to_numeric(df_weather['3pm wind speed (km/h)'], errors= 'coerce')

    remove_missing_value(df_weather)
    
    deduplicate(df_weather)

    drop_outlier_data(df_weather, ['Date'])

    return df_weather