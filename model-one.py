import pandas as pd
import data_cleaning
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

#call out modules to clean datasets
data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

#remove hours and minus
df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'].dt.date)

#calculate daily total demand
df_daily_totaldemand = df_price_demand.groupby(['SETTLEMENTDATE']).sum()

#
df_weather_daily_totaldemand_joined = pd.merge(df_weather, df_daily_totaldemand, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')

for colunm_index in range(1, 21):
    feature_list = []
    for colunm_name in df_weather_daily_totaldemand_joined.columns[colunm_index:21]:
        
        feature_list.append(colunm_name)
        print(feature_list)
 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=1)

    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)

    r2_test = lm.score(X_test, y_test)
    lm.predict(X_test.head())
    y_test.head()