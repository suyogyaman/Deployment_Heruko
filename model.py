# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:16:49 2020

@author: Suyog
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('hiring.csv')

data['experience'].fillna(0,inplace=True)

data['test_score'].fillna(data['test_score'].mean(),inplace=True)

X = data.iloc[:,:3]
y = data.iloc[:,-1]

#Convert words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
                 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

#Model building
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting the model
regressor.fit(X,y)

#Prediction
#int_feature = [int(x) for x in [10,7,8]]
#p = [np.array(int_feature)]
#pred = regressor.predict(p)

#Saving the model to disk
pickle.dump(regressor,open('model.pkl','wb'))

#Loading the model
model = pickle.load(open('model.pkl','rb'))