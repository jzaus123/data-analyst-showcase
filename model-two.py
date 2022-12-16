import pandas as pd
import data_cleaning

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

#call out modules to clean datasets
data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

#remove hours and minus
df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'].dt.date)

#calculate maxi daily price category
df_maxi_daily_price = df_price_demand.groupby(['SETTLEMENTDATE']).agg(pd.Series.mode)

df_weather_price_category_joined = pd.merge(df_weather, df_maxi_daily_price, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')
print(df_weather_price_category_joined)
