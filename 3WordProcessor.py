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
from shutil import copyfile
from os import getcwd
import csv

##
vocab_size = 10000
embedding_dim = 100
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

print(len(stopwords))


##
sentences = []
label = []

with open("Code/Exercise/Resources/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        label.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)

print(len(label))
print(len(sentences))

##
#print(len(sentence))
splitpoint = int(len(sentences) * training_portion)
train_sentences = sentences[:splitpoint]
train_label = label[:splitpoint]
test_sentences = sentences[splitpoint:]
test_label = label[splitpoint:]

print(splitpoint)
print(len(train_sentences))
print(len(train_label))
print(len(test_sentences))
print(len(test_label))

##
token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
token.fit_on_texts(train_sentences)
word_index = token.word_index
vocab_size = len(word_index)

train_sequences = token.texts_to_sequences(train_sentences)
train_pad = pad_sequences(train_sequences, maxlen= max_length, padding=padding_type,truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_pad[0]))

print(len(train_sequences[1]))
print(len(train_pad[1]))

print(len(train_sequences[10]))
print(len(train_pad[10]))
##
test_sequences = token.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

print(test_sequences)
print(test_pad.shape)
##
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(label)

#label_tokenizer = Tokenizer()
#label_tokenizer.fit_on_texts(label)

#train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
#test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_label))

train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
test_label_seq  = np.array(label_tokenizer.texts_to_sequences(test_label))

print(train_label_seq[0])
print(train_label_seq[1])
print(train_label_seq[2])
print(train_label_seq.shape)

print(test_label_seq[0])
print(test_label_seq[1])
print(test_label_seq[2])
print(test_label_seq.shape)
##
print(embedding_dim)
##
model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=vocab_size + 1,output_dim=embedding_dim, input_length= max_length ),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Conv1D(64, 5, activation='relu'),
                             tf.keras.layers.MaxPooling1D(4),
                             #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=[None, 1]),
                             tf.keras.layers.LSTM(64),
                             tf.keras.layers.Dense(32, activation='relu'),
                             tf.keras.layers.Dense(6, activation='softmax')
                             ])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['acc'])
model.summary()

history = model.fit(train_pad, train_label_seq, epochs = 20, batch_size=20, validation_data=(test_pad, test_label_seq),verbose =1)

##
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")

##
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

##
import io

out_v = io.open('Code/Exercise/Output/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('Code/Exercise/Output/meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()