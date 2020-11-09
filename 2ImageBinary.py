##
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
##
import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

##
zip_path = 'Code/Exercise/Resources/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(zip_path, 'r')
zip_ref.extractall('Code/Exercise/Resources/')

zip_ref.close()

##
try:
  os.mkdir('Code/Exercise/Resources/cats_v_dogs')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/training')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/testing')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/training/cats')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/training/dogs')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/testing/cats')
  os.mkdir('Code/Exercise/Resources/cats_v_dogs/testing/dogs')
except OSError:
  pass

##
def split_data(source, train, test, split):
    random_list = random.sample(os.listdir(source), len((os.listdir(source))))
    split_index = int(len(source) * split)
    train_list = random_list[:split_index]
    test_list = random_list[split_index:]
    for item in train_list:
        if os.path.getsize(source + "/" + item) > 0:
            copyfile(source + '/' + item, train + item)

    for item in test_list:
        if os.path.getsize(source + '/' + item) >0:
            copyfile(source + "/" + item, test + item)

##
cat_source = 'Code/Exercise/Resources/cats_and_dogs_filtered/train/cats'
cat_train = 'Code/Exercise/Resources/cats_v_dogs/training/cats/'
cat_test = 'Code/Exercise/Resources/cats_v_dogs/testing/cats/'

dog_source = 'Code/Exercise/Resources/cats_and_dogs_filtered/train/dogs'
dog_train = 'Code/Exercise/Resources/cats_v_dogs/training/dogs/'
dog_test = 'Code/Exercise/Resources/cats_v_dogs/testing/dogs/'

split_size = 0.9
split_data(cat_source, cat_train, cat_test,split_size)
split_data(dog_source, dog_train, dog_test,split_size)

##
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range=0.4,
                                   horizontal_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('Code/Exercise/Resources/cats_and_dogs_filtered/train',
                                                    target_size=(150,150),
                                                    class_mode='binary')

test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory('Code/Exercise/Resources/cats_and_dogs_filtered/validation',target_size=(150,150),class_mode='binary')

##
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3) , activation='relu'),
                                    tf.keras.layers.MaxPooling2D(4),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation= 'relu'),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()
##

history = model.fit(train_generator, epochs = 10, validation_data= test_generator)

##

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)