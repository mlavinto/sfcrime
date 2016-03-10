#
# My entry for the San Francisco Crime Classification problem given in Kaggle, https://www.kaggle.com/c/sf-crime
# Uses a random forest classifier. Takes into account whether a crime occurred in an intersection or not, and time 
# of day is considered as a cyclic variable.
#
# 

import pandas as pd
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
	
# Read the date string and return a dictionary with year, month, day, hour and
# the components of a periodic time vector (for which 23:59 and 0:01 are close to each other)
def read_date(strng):
    arr = strng.split(' ')
    date = [int(el) for el in arr[0].split('-')]   
    time = [int(el) for el in arr[1].split(':')]
    TT = (pi*(time[0] + time[1]/60. + time[2]/3600.)/12.)
    return {'Year':date[0], 'Month':date[1], 'Day':date[2], 'Hour':time[0], 'SinT':sin(TT), 'CosT':cos(TT)} 

# Take the cvs file, modify the date format
def parse_data(data):
    times = pd.DataFrame([read_date(el) for el in data.Dates])
    corner = pd.DataFrame(['/' in addr for addr in data.Address])
    corner.columns = ['Corner']
    return data.join(times).join(corner)

#Define parameters for fitting and do one-shot-encoding for districts and weekdays
columns_to_use = ['Corner', 'SinT', 'CosT', 'Month', 'Year']

def data_to_fit(data): 
    dummies = (pd.get_dummies(data['DayOfWeek']))
    scaled_X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(data.X.reshape(-1,1)), columns=['X'])
    scaled_Y = pd.DataFrame(preprocessing.StandardScaler().fit_transform(data.Y.reshape(-1,1)), columns=['Y'])
    return dummies.join(data[columns_to_use]).join(scaled_X).join(scaled_Y)
    
# Load the train and test sets	
test = parse_data(pd.read_csv('test.csv'))
train = parse_data(pd.read_csv('train.csv'))
print('Data loaded')

# Lists of unique categories, districts and years
categories = sorted(train['Category'].unique())
districts = sorted(train['PdDistrict'].unique())
years = sorted(train['Year'].unique())

#Initiate the classifier model and encode crime labels
model = RandomForestClassifier(n_estimators=10, max_depth=10)
laben = preprocessing.LabelEncoder()
target = laben.fit_transform(train.Category)

#Prep the data sets for fitting
fit_train = data_to_fit(train)
fit_test = data_to_fit(test)

# This piece is for testing different models
X_train, X_test, y_train, y_test = train_test_split(fit_train, target, test_size=0.3, random_state=17)
def run_model():
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)
    return pred
    
#Fit model, make predictions and print output
#Comment this piece when testing
print('Fitting data...')
model.fit(fit_train, target)
print('Making predictions...')
pred = model.predict_proba(fit_test)
result = pd.DataFrame(pred, columns=laben.classes_)
print('Printing...')
result.to_csv('result.csv', index=True, index_label='Id', float_format='%.5e')
