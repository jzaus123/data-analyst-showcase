# Overview

This project serves as a showcase project which demostrates my data analysis skills. 

Two datasets are given containing Victoria's [weather data](weather_data.csv) and [energy price data](price_demand_data.csv).

Two models are built based on the given datasets:
- [Model one](model-one.py) aims to predict the total daily energy usage based on the provided weather dataset. 
- [Model two](model-two.py) aims to predict the maximum daily price category based on the provided weather dataset. 

# Dependecies
- Pandas
- Scikit-learn

# How to run the code
## Install the dependencies

```
python -m pip install pandas
python -m pip install scikit-learn
```

## Run python scripts
```
python model-one.py
python model-two.py
```

# Model Building process 
1. Importing Python libraries
2. Loading the two datasets
3. Applying Data-cleaning module to clean datasets
4. Data preparation - merging two datasets into dataframe for analysing
5. Determining features for producing models - Pearson’s Correlation Coefficient & Mutual Information
6. Defining dependant and independant variables for training and testing data
7. Defining algorithms - Regression for numeric outputs & Classification for categorical outputs
8. Looking for a feature conbimation to compute best model evaluation using R-Squared or Accuracy Score 