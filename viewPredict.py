#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:27:13 2020

@author: seangao
"""

import pandas as pd
import re
import contractions
import spacy 
import en_core_web_sm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Embedding, Dense, Dropout, Flatten, LSTM
import matplotlib.pyplot as plt
import numpy as np
import keras

#LOAD DATA

nt = pd.read_csv('[YOURDATAPATH]')

list(nt.columns) 
nt = nt[['youtube_daily_title', 'youtube_daily_sum_total_views']]
nt.columns = ['title', 'views']

#FIX CONTRACTIONS
nt['title'] = nt['title'].apply(lambda x: contractions.fix(x))

#REMOVE PUNCTUATION
nt['title'] = nt['title'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

#CONVERT TO LOWERCASE
def lowerCase(input_str):
    input_str = input_str.lower()
    return input_str

nt['title'] = nt['title'].apply(lambda x: lowerCase(x))

#LEMMATIZATION
sp = en_core_web_sm.load()

def lemma(input_str):
    s = sp(input_str)
    
    input_list = []
    for word in s:
        w = word.lemma_
        input_list.append(w)
        
    output = ' '.join(input_list)
    return output

nt['title'] = nt['title'].apply(lambda x: lemma(x))

#VECTORIZE
tokenizer = Tokenizer(num_words = 10000, split = ' ')
tokenizer.fit_on_texts(nt['title'].values)

X = tokenizer.texts_to_sequences(nt['title'].values)
X = pad_sequences(X)

num_words = len(tokenizer.word_index) + 1

nt['views'] = nt['views'].apply(lambda x: np.log(x))
y = nt['views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42)

#BUILD THE MODEL
model = Sequential()

model.add(Embedding(num_words, output_dim = 128, 
                    mask_zero = False, input_length = X.shape[1]))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation = 'linear'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', 
              metrics = ['mse'])

model.summary()

#TRAIN THE MODEL
history = model.fit(X_train, y_train, 
                    epochs = 100, batch_size = 32, verbose = 1,
                    validation_data = (X_test, y_test))

#PLOT
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#PREDICT NEW TITLE
import pickle
 
with open('~/tokenizer1.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('~/tokenizer1.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

def predictView(title):
    import math
    
    input_str = title
    input_df = pd.DataFrame(columns=['title', 'view'])
    input_df = input_df.append({'title' : input_str} , ignore_index=True)
    seq = loaded_tokenizer.texts_to_sequences(input_df['title'].values)
    padded = pad_sequences(seq, 16)
    pred = math.exp(model.predict(padded))
    return pred

#TEST

print(predictView("[NEWTITLE]"))