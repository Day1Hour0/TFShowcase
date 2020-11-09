##
import os
import zipfile
import random
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import tensorflow.keras.regularizers as regu
from shutil import copyfile
from os import getcwd
import csv

##
data = open('Code/Exercise/Resources/sonnets.txt').read()

tokenizer = Tokenizer()

corpus = data.lower().split("\n")


tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []

for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)

##
model = tf.keras.models.Sequential([tf.keras.layers.Embedding(total_words, 100, input_length= max_sequence_len - 1),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.LSTM(64),
                                    tf.keras.layers.Dense(total_words/ 2, activation='relu', kernel_regularizer= regu.l2(0.01)),
                                    tf.keras.layers.Dense(total_words, activation = 'softmax')
                                    ])

model.compile(optimizer= 'adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.summary()

##
history = model.fit(predictors, label, epochs = 100, verbose = 1)
##
import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()

##
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)