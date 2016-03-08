#
# My entry for the San Francisco Crime Classification problem given in Kaggle, https://www.kaggle.com/c/sf-crime
# Uses Naive Bayesian classifier. Takes into account whether a crime occurred in an intersection or not, and time 
# of day is considered as a cyclic variable.
#
# 

import pandas as pd
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
	
# Read the date string and return a dictionary with year, month, day and hour
def read_date(strng):
    arr = strng.split(' ')
    date = [int(el) for el in arr[0].split('-')]   
    time = [int(el) for el in arr[1].split(':')]
    pertime = sin(pi*(time[0] + time[1]/60. + time[2]/3600.)/24.)
    return {'Year':date[0], 'Month':date[1], 'Day':date[2], 'Hour':time[0], 'Periodic':pertime} 

# Take the cvs file, modify the date format
def parse_data(data):
    times = pd.DataFrame([read_date(el) for el in data.Dates])
    corner = pd.DataFrame(['/' in addr for addr in data.Address])
    corner.columns = ['Corner']
    return data.join(times).join(corner)

# Load the train and test sets	
test = parse_data(pd.read_csv('test.csv'))
train = parse_data(pd.read_csv('train.csv'))

print('Data loaded')

# Lists of unique categories, districts and years
categories = sorted(train['Category'].unique())
districts = sorted(train['PdDistrict'].unique())
years = sorted(train['Year'].unique())

#Initiate the classifier model and encode crime labels
#Bernoulli Naive Bayes seems the fastest and most stabile out of the things I tried
model = BernoulliNB()
laben = preprocessing.LabelEncoder()
target = laben.fit_transform(train.Category)

#Define parameters for fitting and do one-shot-encoding for districts and weekdays
fit_data = pd.get_dummies(train['PdDistrict']).join(pd.get_dummies(train['DayOfWeek'])).join(train[['X', 'Y', 'Corner', 'Periodic', 'Month', 'Year']])
fit_test = pd.get_dummies(test['PdDistrict']).join(pd.get_dummies(test['DayOfWeek'])).join(test[['X', 'Y', 'Corner', 'Periodic', 'Month', 'Year']])

# This piece is for testing different models
X_train, X_test, y_train, y_test = train_test_split(fit_data, target, test_size=0.3, random_state=17)
def run_model():
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)
    return pred
    
#Fit model, make predictions and print output
#Comment this piece when testing
print('Fitting data...')
model.fit(fit_data, target)
print('Making predictions...')
pred = model.predict_proba(fit_test)
result = pd.DataFrame(pred, columns=laben.classes_)
print('Printing...')
result.to_csv('result.csv', index=True, index_label='Id')
