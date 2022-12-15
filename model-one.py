import pandas as pd
import data_cleaning

pd.set_option('display.max_rows', 500) 

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'].dt.date)
df_daily_totaldemand = df_price_demand.groupby(['SETTLEMENTDATE']).sum()

df_weather_daily_totaldemand_joined = pd.merge(df_weather, df_daily_totaldemand, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')
print(df_weather_daily_totaldemand_joined.shape)


