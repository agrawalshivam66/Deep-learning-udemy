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

#adding second convolutional layer
#input shape is already given
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3 Flattening
classifier.add(Flatten())

#step 4 full connection
classifier.add(Dense(units=128, activation= 'relu'))
#output layer
classifier.add(Dense(units=1, activation= 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Part 2 fitting the CNN to our images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

# part 3 single prediction
import numpy as np
from keras.preprocessing import image 
test_image = image.load_img('dataset\single_prediction\cat_or_dog_1.jpg',target_size=( 64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'