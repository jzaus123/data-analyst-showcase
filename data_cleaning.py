import pandas as pd
import numpy as np

def clean_price_demand_dataset(df_price_demand):
    
    df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'], format='%d/%m/%Y %H:%M', errors='ignore')

    #check amount of NaN value in each colunm and decide if NaN value is dropped off or filled in aggregated values
    for colunm_name in df_price_demand.columns:
        price_missing_num = df_price_demand[colunm_name].isnull().sum()

        if price_missing_num != 0:
            print(f'{colunm_name} has {price_missing_num} NaN')

    #based on above check there is a small amount of NaN values so we can just drop them off
    df_price_demand.dropna(axis = 0, inplace = True)

    # TODO Deduplicate your data - no duplicated data by checking via df.duplicated()

    #Filter out data outliers
    #use IQR (Inter Quartile Range)-IQR = Quartile3 – Quartile1
    for colunm_name in df_price_demand.columns:
        if df_price_demand[colunm_name].dtypes == 'object' or colunm_name == 'SETTLEMENTDATE':
            continue
        Q1 = np.percentile(df_price_demand[colunm_name], 5, method = 'midpoint')
        Q3 = np.percentile(df_price_demand[colunm_name], 95, method = 'midpoint')

        IQR = Q3 - Q1      

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        df_upper_bound = df_price_demand.loc[df_price_demand[colunm_name] > upper_bound]
        df_lower_bound = df_price_demand.loc[df_price_demand[colunm_name] < lower_bound]

        df_price_demand.drop(df_upper_bound, inplace = True, errors = 'ignore')
        df_price_demand.drop(df_lower_bound, inplace = True, errors = 'ignore')

    return df_price_demand
  
def clean_weather_dataset(df_weather):

    #convert data types to datetime
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d/%m/%Y', errors='ignore')

    #convert object to float 
    df_weather['9am wind speed (km/h)'] = pd.to_numeric(df_weather['9am wind speed (km/h)'], errors='coerce')
    df_weather['3pm wind speed (km/h)'] = pd.to_numeric(df_weather['3pm wind speed (km/h)'], errors= 'coerce')

    #Deal with missing data
    total_missing_num = 0
    for colunm_name in df_weather.columns:
        weather_missing_num = df_weather[colunm_name].isnull().sum()
        total_missing_num += weather_missing_num

    #drop off NaN values
    df_weather.dropna(axis= 0, inplace=True)

    #check NaN value 
    df_weather.isnull().values.any()

    return df_weather