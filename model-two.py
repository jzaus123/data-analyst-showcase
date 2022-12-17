import pandas as pd
import data_cleaning
import math
import copy
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing

df_price_demand = pd.read_csv('price_demand_data.csv')
df_weather = pd.read_csv('weather_data.csv')

#call out modules to clean datasets
data_cleaning.clean_price_demand_dataset(df_price_demand)
data_cleaning.clean_weather_dataset(df_weather)

#remove hours and minus
df_price_demand['SETTLEMENTDATE'] = pd.to_datetime(df_price_demand['SETTLEMENTDATE'].dt.date)

#calculate maxi daily price category
df_weather_price_category_joined = df_price_demand.groupby(['SETTLEMENTDATE']).agg(pd.Series.mode)

df_weather_price_category_joined = pd.merge(df_weather, df_weather_price_category_joined, how = 'inner', left_on = 'Date', right_on = 'SETTLEMENTDATE')

#remove 'REGION' colunm 
df_weather_price_category_joined.drop('REGION', axis = 1, inplace = True)

#use square root rule to compute numbers of bin
bin_number = round(math.sqrt(len(df_weather_price_category_joined['Date'])))

#convert object variables to numeric variables in order to calculate distance metric DOES not support categorical variables
#use Binning to convert numerical to categorical
NMI_dict = {}
for column_name in df_weather_price_category_joined.columns:
    NMI = 0
    if column_name in ['PRICECATEGORY', 'Date']:
        continue
    elif df_weather_price_category_joined[column_name].dtypes == 'object':
        df_weather_price_category_joined[column_name] = pd.factorize(df_weather_price_category_joined[column_name])[0] 
        column = pd.cut(df_weather_price_category_joined[column_name], bin_number)
        NMI = normalized_mutual_info_score(column, df_weather_price_category_joined['PRICECATEGORY'], average_method='min')
        NMI_dict[column_name] = NMI
    
    else:
        column = pd.cut(df_weather_price_category_joined[column_name], bin_number)
        NMI = normalized_mutual_info_score(column, df_weather_price_category_joined['PRICECATEGORY'], average_method='min')
        NMI_dict[column_name] = NMI

#find out features with highest NMI
selected_feature_count = 20
features_selection = sorted(NMI_dict.items(), key = lambda x:x[1], reverse = True)[:selected_feature_count]

features_selection = [feature_name[0] for feature_name in features_selection]
best_accuracy_score = 0
best_feature_list = []
for feature_index in range(0, len(features_selection)):
    features_list = []
    for features in features_selection[feature_index:]:
        features_list.append(features)

        feature_data = df_weather_price_category_joined[features_list]
        targetlabel = df_weather_price_category_joined['PRICECATEGORY']
        features_list_train, features_list_test, targetlabel_train, targetlabel_test = train_test_split(feature_data, targetlabel, test_size=0.2, random_state=42)

        #modelling - K-Nearest-Neighbors 
        scaler = preprocessing.StandardScaler().fit(features_list_train)
        features_train = scaler.transform(features_list_train)
        features_test = scaler.transform(features_list_test)

        knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
        knn.fit(features_list_train, targetlabel_train)

        predictions = knn.predict(features_list_test)
        acc_score = accuracy_score(targetlabel_test, predictions)

        if acc_score > best_accuracy_score:
            best_accuracy_score = acc_score
            best_feature_list = copy.deepcopy(features_list)

print(f'Best r2 test score {best_accuracy_score} with features combination {best_feature_list}')