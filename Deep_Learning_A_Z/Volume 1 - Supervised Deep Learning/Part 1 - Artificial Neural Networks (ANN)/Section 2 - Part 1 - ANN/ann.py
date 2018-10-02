# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:45:01 2018

@author: Shivam-PC
"""

# Data Preprocessing

# Importing the libraries
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

#part 2 making the ANN

#import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# initializing ANN
classifier = Sequential()

# Adding input layers and first hidden layers with 
    #ounits-(input layers+output layers)/2,
    #kernel_initializer- for initializing uniform weights nearly to zero
    #activation funcltion 'relu'- rectifier function
    #input_dim- nnumber of input layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#dropout to remove overfitting
#0.1% neurons dropout 
classifier.add(Dropout(rate=0.1))

# Adding second hidden layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#dropout to remove overfitting
#0.1% neurons dropout 
classifier.add(Dropout(rate=0.1))

# Adding output layer
    #only 0 or 1 output needed so output_dim=1
    #activation funcltion 'sigmoid'- sigmoid function-- gives probability, only two catagory
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#compiling the ANN
    #optimiser- aglo to find the best weight- 'adam'- statistic gradient decent algo
    #loss - loss function to find the loss like sum of square loss function. logarithmatic loss for binary outcome
    #metrics- accuracy used to improve weight to improve the model performance
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to training set
    #X_train, y_train - training data and labels
    #batch_size-number of observation after which we update weights 
    #reinforcement learning- updateweights afer each observation
    #batch learning- update weights after batch of observation 
    #epochs- number of epochs- number of times we want to repeat whole process
classifier.fit(X_train, y_train, batch_size=10, epochs=100 ) 

#making predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#part 3
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

#part 4
# Evaluating, improving and tuning ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# builds the ANN classifier and retuns classifier object
def build_classifier():
    # initializing ANN
    classifier = Sequential()
    # Adding input layers and first hidden layers 
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    # Adding second hidden layers
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)

#cv is number of folds we want
#n_jobs=number of cpu we want- -1 will use all the CPUs
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1 )
mean = accuracies.mean()
variance = accuracies.std()

#imporving the ANN
#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense

# builds the ANN classifier and retuns classifier object
def build_classifier(optimizer):
    # initializing ANN
    classifier = Sequential()
    # Adding input layers and first hidden layers 
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    # Adding second hidden layers
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
#grid search tunes epochs and batch size
classifier = KerasClassifier(build_fn=build_classifier)

#parameter dicionary with combination of all the parameters to optimize
parameters = {'batch_size':[25,32],
             'nb_epoch':[100,500],
             'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs=None)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_









