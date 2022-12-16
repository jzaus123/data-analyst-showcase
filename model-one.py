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

df_weather_daily_totaldemand_joined = pd.merge(df_weather, df_daily_totaldemand, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')

#convert object variables to numeric variables in order to calculate Pearson's coorelation coefficient
#features selection by appling Pearson's coorelation coefficient
pearson_coorelation_dict = {}
for colunm_name in df_weather_daily_totaldemand_joined.columns:
    pearson_corr = 0
    if colunm_name in ['TOTALDEMAND', 'Date']:
        continue
    elif df_weather_daily_totaldemand_joined[colunm_name].dtypes == 'object':
        df_weather_daily_totaldemand_joined[colunm_name] = pd.factorize(df_weather_daily_totaldemand_joined[colunm_name])[0] 
        pearson_corr = df_weather_daily_totaldemand_joined[colunm_name].corr(df_weather_daily_totaldemand_joined['TOTALDEMAND'])
        pearson_coorelation_dict[colunm_name] = abs(pearson_corr)
    
    else:
        pearson_corr = df_weather_daily_totaldemand_joined[colunm_name].corr(df_weather_daily_totaldemand_joined['TOTALDEMAND'])
        pearson_coorelation_dict[colunm_name] = abs(pearson_corr)       

#find out top six features with highest Pearson coorealtion coefficient
features_selection = sorted(pearson_coorelation_dict.items(), key = lambda x:x[1], reverse = True)[:6]

features_selection = [feature_name[0] for feature_name in features_selection]
for feature_index in range(0, 7):
    features_list = []
    for features in features_selection[feature_index:6]:
        features_list.append(features)

        feature_data = df_weather_daily_totaldemand_joined[features_list]
        targetlabel = df_weather_daily_totaldemand_joined['TOTALDEMAND']
        features_list_train, features_list_test, targetlabel_train, targetlabel_test = train_test_split(feature_data, targetlabel, test_size=0.2, random_state=42)

        lm = linear_model.LinearRegression()
        model = lm.fit(features_list_train, targetlabel_train)
        print(f'{features_list}{lm.coef_}, {lm.intercept_}')
        r2_test = lm.score(features_list_test, targetlabel_test)
        print(f'{features_list},{r2_test}')
        print(lm.predict(features_list_test.head()))
        print(targetlabel_test.head())