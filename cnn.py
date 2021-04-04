# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator


#If you're using Google Colab, I suggest uploading the folders 
# with the images on Google Drive and using them in Colab with this code
###### UNCOMMENT THE FOLLOWING LINES IF YOU ARE USING COLAB ######
# from google.colab import drive
# drive.mount('/content/gdrive')
# !ls '/content/gdrive'  # take a look at the Google Drive content
###### END OF INSTRUCTIONS FOR COLAB USERS ######


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', # /content/gdrive/MyDrive/dataset/training_set for COLAB users
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', # /content/gdrive/MyDrive/dataset/test_set for COLAB users
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

training_set.class_indices

# if you have the latest version of tensorflow, the fit_generator is deprecated.
# you should use the fit method (i.e., hist=classifier.fit).
# if you do not have the last version you must use fit_generator
hist=classifier.fit_generator(training_set,
                         steps_per_epoch = 250, # if batch_size=1 -> steps_per_epoch=8000
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 63)   


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# if you have the last version of tensorflow, the predict_generator is deprecated.
# you should use the predict method.
# if you do not have the last version, you must use predict_generator
Y_pred = classifier.predict_generator(test_set, 63) # ceil(num_of_test_samples / batch_size)
Y_pred = (Y_pred>0.5)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, Y_pred))
print('Classification Report')
target_names = ['Cats', 'Dogs']
print(classification_report(test_set.classes, Y_pred, target_names=target_names))
