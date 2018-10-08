# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# keras neuron network takes numpy array input so to create numpy array.
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# we will use normalization for all sigmoid functions
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Converting list to array for neuron input
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping to add new dimention
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1) )

# part 2 Building the RNN

# importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and adding Droupout regularisation

#units - number of cells/neurons LSTM
#return sequence - true - several layers can be added more
#input shape - shape of input
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and Droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and Droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and Droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))#no return
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)














