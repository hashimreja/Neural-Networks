# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:25:40 2019

@author: HASHIM REJA SYED
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform',input_dim=11))
classifier.add(Dropout(0.0))
classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))
#compiling the network 
#learning rate 0.001
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=32,epochs=75)

y_pred = classifier.predict(X_test)
y_pred = y_pred >0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


#evaluating the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def building_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dropout(0.0))
    classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = building_classifier,epochs = 75,batch_size=32)
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10)

accuracies.mean()
accuracies.std()

#tuninig the ann 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def building_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dropout(0.0))
    classifier.add(Dense(units = 32,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform')) 
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = building_classifier)
parameters = {'epochs':[70,300],'batch_size':[25,32],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
 grid_search = grid_search.fit(X_train,y_train)

best_param = grid_search.best_params_
best_score = grid_search.best_score_



