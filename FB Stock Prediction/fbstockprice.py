# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:19:50 2019

@author: Syed Hashim Reja
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing the dataset
dataset = pd.read_csv('fbstocks.csv')
#selecting the feature
X = dataset.iloc[0:1665,1:2].values
#scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range =(0,1))
X = sc.fit_transform(X)
##organizing the dataset 
X_train = []
y_train = []
for i in range(60,1665):
    X_train.append(X[i-60:i,0])
    y_train.append(X[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#importing the libraries for rnn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
##Architecture of the network
regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences=True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
#compiling the network
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(X_train,y_train,batch_size = 32,epochs = 100)
#completed training our model with an mse of 0.0010
#prediction set
X_test1 = []
for i in range(1645,1665):
    X_test1.append(X[i-60:i,0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1,(X_test1.shape[0],X_test1.shape[1],1))
#validation set   
valset = []
for i in range(1665,1685):
    valset.append(dataset['open'][i])
valset = np.array(valset)
valset = valset.reshape(-1,1)
valset = sc.transform(valset)
#visualizing the validation set
plt.plot(range(0,len(valset)),valset)
##predicting the outcome
predicted_stock_price = regressor.predict(X_test1)
#visualizing the results
plt.figure(figsize =(10,6))
plt.plot(range(0,len(predicted_stock_price)),predicted_stock_price)
plt.plot(range(0,len(valset)),valset)
#saving the model
regressor.save('fbmodel')
regressor.save_weights('fbweight.h5')
regressor.summary()
