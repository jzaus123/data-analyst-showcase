import pandas as pd

#read datasets
df_weather = pd.read_csv('weather_data.csv')
df_price_demand = pd.read_csv('price_demand_data.csv') 

#glancing data types 
# print(df_weather.dtypes)
# print(df_price_demand.dtypes)

#converting data types to datetime
df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d%m%Y', errors='ignore')
df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'], format='%d%m%Y:%H:%M:%S', errors='ignore')

#converting object to float 
df_weather['9am wind speed (km/h)'] = pd.to_numeric(df_weather['9am wind speed (km/h)'], errors='coerce')
df_weather['3pm wind speed (km/h)'] = pd.to_numeric(df_weather['3pm wind speed (km/h)'], errors= 'coerce')

#Step 1: Remove irrelevant data


#Step 2: Deduplicate your data
# Step 3: Fix structural errors
# Step 4: Deal with missing data
# Step 5: Filter out data outliers
# Step 6: Validate your data