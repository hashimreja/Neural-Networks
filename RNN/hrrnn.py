# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:17:55 2019

@author: Hashim Reja Syed
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

#initializing the x 
X = dataset_train.iloc[:,1:2].values

#scaling the feature matrix
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#organizing the data to feed in the neural network
#taking 60 previous inputs
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(X[i-60:i,0])
    y_train.append(X[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)
#reshaping the x_train and y_Train
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#building the recurrent network lstm model
#importing the libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
#initializing the network
regressor = Sequential()
#adding lstm layers and dropouts
#1st layer
regressor.add(LSTM(units = 50,return_sequences= True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#2nd layer
regressor.add(LSTM(units = 50,return_sequences= True))
regressor.add(Dropout(0.2))
#3rd layer
regressor.add(LSTM(units = 50,return_sequences= True))
regressor.add(Dropout(0.2))
#4th layer
regressor.add(LSTM(units = 50,return_sequences= False))
regressor.add(Dropout(0.2))
#final layer
regressor.add(Dense(units = 1))
#compile stage
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
#fitting the algortihm
history = regressor.fit(X_train,y_train,batch_size = 32,epochs = 100)

#organizing the test set for predictions

total_data = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = total_data[len(total_data)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicited_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicited_stock_price)

#visualization of the data
plt.plot(y,color = 'red',label = 'orginal')
plt.plot(predicted_stock_price,color = 'red',label = 'predicted')
plt.xlabel('date')
plt.ylabel('open rate')
plt.legend()
plt.show()

#saving the model 

regressor.save('kirillmodel')
regressor.save_weights('weights.h5')
regressor.summary()




