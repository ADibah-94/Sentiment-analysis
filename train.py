# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:51:34 2022

This train.py python trains the sentiment to determine if the revision 
is positive or negative

@author: HP
"""

import pandas as pd
from sentiment_analysis_modules import ExploratoryDataAnalysis, ModelCreation
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import  train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
TOKEN_SAVE_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'logs') 

# EDA
# Step 1) Import Data

df = pd.read_csv(URL) 
review = df['review']
sentiment = df['sentiment']

# Step 2) Data cleaning
# Remove tags

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review) # to remove tags
review = eda.lower_split(review) # to convert to lower case and split

# Step 3) Features Selection
# step 4) Data vectorization
 
review = eda.sentiment_tokenizer(review, TOKEN_SAVE_PATH)
review = eda.sentiment_pad_sequence(review)

# Step 5) Preprocessing
# One hot encoder

one_hot_encoder = OneHotEncoder(sparse=False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment,
                                                            axis=-1))
# To calculate the number of total categories
nb_categories = len(np.unique(sentiment))


# X = review(feature) , Y = sentiment(target)
# split train test

X_train,X_test,y_train,y_test = train_test_split(review, 
                                                 sentiment, 
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

# from here you will know that [0,1] is positive and [1,0] is negative
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))

mc = ModelCreation()
num_words = 10000
model = mc.lstm_layer(num_words, nb_categories)

#%% Callbacks

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


#%% Compile & Model fitting

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(X_train,y_train,
          epochs=5, 
          validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback])

#%% Model Evaluation
# Preallocation of of memory approach

predicted_advanced = np.empty([len(X_test), 2])
for index, test in enumerate(X_test):
    predicted_advanced[index,:] =  model.predict(np.expand_dims(test,axis=0))

#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report.metrics(y_true,y_pred)

#%% Model Deployment

model.save(MODEL_SAVE_PATH)

