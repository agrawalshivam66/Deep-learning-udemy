# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialinsing the CNN
classifier = Sequential()

# step 1 - Convolution
#input shape = 3- colors, 64 X 64 pixel image
#32-nb filter-number of feature maps
#3,3 number of rows and columns in feature detector

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#step 2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3 Flattening
classifier.add(Flatten())

#step 4 full connection
classifier.add(Dense(units=128, activation= 'relu'))
#output layer
classifier.add(Dense(units=1, activation= 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

