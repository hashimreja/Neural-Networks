# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:25:44 2019

@author: Hashim reja Syed
"""
#importing the libraries
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
classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform',input_dim=11))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
classifier.add(Dropout(0.0))
classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
classifier.add(Dropout(0.0))
classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
classifier.add(Dropout(0.0))
classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='he_uniform'))
#compiling the network 
#adam = keras.optimizers.Adam(lr = 0.001)
sgd = keras.optimizers.SGD(lr =0.01,momentum =0.0)
classifier.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
history = classifier.fit(X_train,y_train,batch_size=15,epochs=200,validation_data = (X_test,y_test))
#predicting
y_pred = classifier.predict(X_test)
y_pred = y_pred >0.5
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

## visualizing the error
plt.figure(figsize=(11,8)) 
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('error')
plt.xlabel('epochs')
plt.legend(['train','test'])
plt.show()

#visualizing the accuracy
plt.figure(figsize=(11,8)) 
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','test'])
plt.show()


#tuninig the model
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def tunemodel():
    classifier = Sequential()
    classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform',input_dim=11))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
    classifier.add(Dropout(0.0))
    classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
    classifier.add(Dropout(0.0))
    classifier.add(Dense(units = 20,activation='softplus',kernel_initializer='he_uniform'))
    classifier.add(Dropout(0.0))
    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='he_uniform'))
    optimizers = keras.optimizers.SGD(lr = 0.01,momentum = 0.0)
    classifier.compile(optimizer=optimizers,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#step1 evaluating the epochs and batch size
'''
classifier = KerasClassifier(build_fn = tunemodel)
epochs = [200,300]
batch_size = [10,15]
parameters = dict(epochs = epochs,batch_size = batch_size)
gridsearch = GridSearchCV(estimator =classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)
grid_search = gridsearch.fit(X_train,y_train)
params = grid_search.best_params_
score = grid_search.best_score_
'''
#step2 evaluating the optimization algorithm
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs =200,batch_size = 15)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
parameters = dict(optimizer = optimizer)
gridsearch = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)

 grid_search = gridsearch.fit(X_train,y_train)

param = grid_search.best_params_
score = grid_search.best_score_
'''
#step3 evaluating the learning rate and momentum
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs=200,batch_size=15)
learn_rate = [0.1,0.01,0.001]
momentum = [0.0,0.2,0.4]
parameters = dict(learn_rate = learn_rate,momentum = momentum)
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)
grid_search = gridsearch.fit(X_train,y_train)

params = grid_search.best_params_
score = grid_search.best_score_
'''
#step4 evaluating the weights - 86.1
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs=200,batch_size=15)
myweights = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
parameters = dict(myweights = myweights)
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)
grid_search = gridsearch.fit(X_train,y_train)

params = grid_search.best_params_
score = grid_search.best_score_
'''
#step5 evaluating the activation function 86.2
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs=200,batch_size=15)
myact = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
parameters = dict(myact = myact)
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)
grid_search = gridsearch.fit(X_train,y_train)

params = grid_search.best_params_
score = grid_search.best_score_
'''
#later test it hashim
'''
mean = grid_search.cv_results['mean_test_score'
std = grid_search.cv_results['std_test_score']
parameter = grid_search.cv_reults['params']

for mean,std,parameter in zip(mean,std,parameter):
    print('{}{}with{}'.format(mean,std,parameter))
'''
#step6 evaluating the neurons
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs=200,batch_size=15)
neurons = [10,20,30]
parameters = dict(neurons = neurons)
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 3)
grid_search = gridsearch.fit(X_train,y_train)
params = grid_search.best_params_
score = grid_search.best_score_

'''
#step7 evaluating the drop out rate 
'''
classifier = KerasClassifier(build_fn = tunemodel,epochs=200,batch_size=15)
drate = [0.0,0.1,0.2,0.3]
parameters = dict(drate = drate)
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 3)
grid_search = gridsearch.fit(X_train,y_train)

params = grid_search.best_params_
score = grid_search.best_score_
'''
#evaluating the model finally our graphs resulted pretty awesome by the way
'''
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn = tunemodel,batch_size = 15,epochs = 200)

accuracies = cross_val_score(estimator = classifier,
                             X = X_train,y = y_train,
                             cv = 10)
acc_mean = accuracies.mean()
std = accuracies.std()
'''


