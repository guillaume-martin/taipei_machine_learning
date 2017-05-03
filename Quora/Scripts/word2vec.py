#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:50:24 2017

@author: guillaume
@purpose: train a word2vec model using all of Quora questions
"""

# import packages
import pandas as pd
import logging
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import word2vec

#%%

# set variables
DATA_DIR = '../Data/'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

#%%

def question_to_words(question, remove_stopwords=False, stem_words=False):
    ''' transform a question into a list of lowercase words
    parameters
    ----------
    question    string
        the question to be tokenized
    
    remove_stopwords    boolean
        True if the stop words need to be removed. False otherwise
        
    stem_words    boolean
        True if words need to be reduced to stem. False otherwise
        
    return
    ------
    a list of words
    '''
    
    # convert to lowercase
    question = question.lower()
    
    # remove punctuation and split
    question = re.findall(r'\w+', question, flags = re.UNICODE)
    
    # remove stop words
    words=[]
    if remove_stopwords:
        stops = stopwords.words('english')
        words = [word for word in question if not word in stops]
                
    # stem words
    if stem_words:
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(word) for word in words]
        
    if len(words) > 0:
        return words
    else:
        return question
    
#%%

# We import both the training and the test set.
df_train = pd.read_csv(DATA_DIR+TRAIN_DATA_FILE)
df_train = df_train[['question1','question2']]
df_test = pd.read_csv(DATA_DIR + TEST_DATA_FILE)
df_test = df_test[['question1','question2']]

# We merge the 2 dataframes
df = pd.concat([df_train, df_test])

# We create lists of words for each question. 
# To train Word2Vec, it is better to leave stop words
df['words1'] = df['question1'].apply(lambda x: question_to_words(str(x)))
df['words2'] = df['question2'].apply(lambda x: question_to_words(str(x)))
df.head(20)

# we parse all questions into a list of sentences
sentences = []
sentences += df['words1'].tolist()
sentences += df['words2'].tolist()

# setup word embedding variables
num_features = 300
min_word_count = 40 # mini freq for words to be kept
num_workers = 4
context = 20
downsampling = 1e-3
model_name = 'w2v_' + str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# learn the vocabulary
model = word2vec.Word2Vec(sentences, 
                          workers = num_workers,
                          size = num_features,
                          min_count = min_word_count,
                          window = context, 
                          sample = downsampling)


model.save(model_name)
