import pandas as pd
import data_cleaning

pd.set_option('display.max_rows', 500) 

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

df_daily_totaldemand = df_price_demand['TOTALDEMAND'].groupby(df_price_demand['SETTLEMENTDATE'].dt.to_period('D')).sum()

df_weather_daily_totaldemand_joined = pd.merge(df_weather, df_daily_totaldemand, how = 'outer', left_index=True, right_index=True)


