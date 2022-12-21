import pandas as pd
import data_cleaning
import copy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

#call out modules to clean datasets
data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

#remove hours and mins
df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'].dt.date)

#calculate daily total demand
df_daily_totaldemand = df_price_demand.groupby(['SETTLEMENTDATE']).sum()

df_weather_daily_totaldemand_joined = pd.merge(df_weather, df_daily_totaldemand, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')

#convert object variables to numeric variables in order to calculate Pearson's correlation coefficient
#feature selection by appling Pearson's correlation coefficient
pearson_correlation_dict = {}
for column_name in df_weather_daily_totaldemand_joined.columns:
    if column_name in ['TOTALDEMAND', 'Date']:
        continue
    
    if df_weather_daily_totaldemand_joined[column_name].dtypes == 'object':
        df_weather_daily_totaldemand_joined[column_name] = pd.factorize(df_weather_daily_totaldemand_joined[column_name])[0] 

    pearson_corr = df_weather_daily_totaldemand_joined[column_name].corr(df_weather_daily_totaldemand_joined['TOTALDEMAND'])
    pearson_correlation_dict[column_name] = abs(pearson_corr)   

#find out features with highest Pearson correaltion coefficient
selected_feature_count = 6
feature_selection = sorted(pearson_correlation_dict.items(), key = lambda x:x[1], reverse = True)[:selected_feature_count]

feature_selection = [feature_name[0] for feature_name in feature_selection]
best_r2_test_score = 0
best_feature_list = []
for feature_index in range(0, len(feature_selection)):
    feature_list = []
    for feature in feature_selection[feature_index:]:
        feature_list.append(feature)

        feature_data = df_weather_daily_totaldemand_joined[feature_list]
        target_data = df_weather_daily_totaldemand_joined['TOTALDEMAND']
        features_train, features_test, target_train, target_test = train_test_split(feature_data, target_data, test_size=0.2, random_state=42)

        #modelling - linear regression
        lm = linear_model.LinearRegression()
        model = lm.fit(features_train, target_train)
        r2_test_score = lm.score(features_test, target_test)

        if r2_test_score > best_r2_test_score:
            best_r2_test_score = r2_test_score
            best_feature_list = copy.deepcopy(feature_list)

print(f'Best r2 test score {best_r2_test_score} with features combination {best_feature_list}')
