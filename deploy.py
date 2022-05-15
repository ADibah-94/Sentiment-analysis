# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:25 2022

@author: HP
"""

from tensorflow.keras.models import load_model
import os
import json
from sentiment_analysis_modules import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#%% Model loading

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')

sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% tokenizer loading

JSON_PATH =os.path.join(os.getcwd(), 'tokenizer_data.json')

with open(JSON_PATH,'r') as json_file:
    token = json.load(json_file)

#%% EDA

# Step 1) loading of data

#new_review = ['I think the first one hour is interesting but \
#    a second half of movie is boring. This movie just wasted my precious\
#        time and hard earned money. This movie should be banned to avoid\
#            time being wasted.<br \>']

new_review = [list(input('Review about the movie'))]
            
# Step 2) to clean the data

eda = ExploratoryDataAnalysis()
remove_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# Step 3) Features Selection
# step 4) Data preprocessing

# to vectorize the new review
# to feeds tokens into keras
loaded_tokenizer = tokenizer_from_json(token)

#  to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequence(new_review)
new_review = eda.sentiment_pad_sequences(new_review)

#%% model prediction

outcome = sentiment_classifier.predict(np.predict_dims(new_review,axis=-1))
sentiment_dict ={1:'positive', 0:'negative'}
print('this review is ' + sentiment_dict[np.argmax(outcome)])

# positive suppose to be = [0,1]
# negative suppose to be = [1,0]

 






