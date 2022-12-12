import pandas as pd
import numpy as np

def clean_price_demand_dataset(df_price_demand):
    #convert data types to datetime
    df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'], format='%d/%m/%Y %H:%M', errors='ignore')

    #Step 4: Deal with missing data
    price_missing_num = 0
    for colunm_name in df_price_demand.columns:
        price_missing_num = df_price_demand[colunm_name].isnull().sum()

    #drop off NaN values
    df_price_demand.dropna(axis= 0, inplace=True)
    df_price_demand.isnull().values.any()

    # Step 5: Filter out data outliers
    #use IQR (Inter Quartile Range)-IQR = Quartile3 – Quartile1
    for colunm_name in df_price_demand.columns:
        if df_price_demand[colunm_name].dtypes == 'object' or colunm_name == 'SETTLEMENTDATE':
            continue
        Q1 = np.percentile(df_price_demand[colunm_name], 5, method = 'midpoint')
        
        Q3 = np.percentile(df_price_demand[colunm_name], 95, method = 'midpoint')

        IQR = Q3 - Q1      

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        df_upper_bound = df_price_demand.loc[df_price_demand[colunm_name] < upper_bound]
        df_lower_bound = df_price_demand.loc[df_price_demand[colunm_name] > lower_bound]

    return df_price_demand
  
def clean_weather_dataset(df_weather):

    #convert data types to datetime
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d/%m/%Y', errors='ignore')

    #convert object to float 
    df_weather['9am wind speed (km/h)'] = pd.to_numeric(df_weather['9am wind speed (km/h)'], errors='coerce')
    df_weather['3pm wind speed (km/h)'] = pd.to_numeric(df_weather['3pm wind speed (km/h)'], errors= 'coerce')

    #Step 4: Deal with missing data
    total_missing_num = 0
    for colunm_name in df_weather.columns:
        weather_missing_num = df_weather[colunm_name].isnull().sum()
        total_missing_num += weather_missing_num

    #drop off NaN values
    df_weather.dropna(axis= 0, inplace=True)

    #check NaN value 
    df_weather.isnull().values.any()

    # Step 5: Filter out data outliers
    #use IQR (Inter Quartile Range)-IQR = Quartile3 – Quartile1
    for colunm_name in df_weather.columns:
        if df_weather[colunm_name].dtypes == 'object' or colunm_name == 'Date':
            continue
        Q1 = np.percentile(df_weather[colunm_name], 5, method = 'midpoint')
        
        Q3 = np.percentile(df_weather[colunm_name], 95, method = 'midpoint')

        IQR = Q3 - Q1      

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        df_upper_bound = df_weather.loc[df_weather[colunm_name] < upper_bound]
        df_lower_bound = df_weather.loc[df_weather[colunm_name] > lower_bound]
    
    return df_weather