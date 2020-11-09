##
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.utils as ku
from os import getcwd
##
def get_data(filename):
    labels = []
    images = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        lineNo = 0
        for line in reader:
            if lineNo == 0:
                lineNo += 1
            else:
                #print(line)
                #item = line.split(',')
                #print(item)
                labels.append(line[0])
                #print(item)
                images.append(np.array(line[1:]).reshape(28,28))

    labels = np.array(labels).astype(float)
    images = np.array(images).astype(float)

    return images, labels

##

train_img, train_label = get_data('Code/Exercise/Resources/sign_mnist_train.csv')
test_img, test_label = get_data('Code/Exercise/Resources/sign_mnist_test.csv')

print(train_img.shape)
print(test_img.shape)
##
train_img = np.expand_dims(train_img, axis = -1)
test_img = np.expand_dims(test_img, axis = -1)
print(train_img.shape)

##
train_label = ku.to_categorical(train_label, num_classes=26)
test_label = ku.to_categorical(test_label, num_classes=26)
##
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range= 40,
                                   fill_mode='nearest',
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.2)

test_datagen=ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow(train_img, train_label)
test_generator = test_datagen.flow(test_img, test_label)

##
print(len(set(train_label)))

##
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    #tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
                                    #tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(26, activation= 'softmax')])

##
from tensorflow.keras.optimizers import RMSprop
##
model.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['acc'])

model.summary()

##
history = model.fit(train_generator, epochs = 10,steps_per_epoch = len(train_img) / 32,batch_size = 32, validation_data= test_generator)

##
