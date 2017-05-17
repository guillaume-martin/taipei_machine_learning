#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 08:34:29 2017

@author: guillaume
@ purpose: save a list of vectors for each questions
"""

# import packages
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec


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

# set variables
DATA_DIR = '../Data/'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

# load the data
df = pd.read_csv(DATA_DIR + TRAIN_DATA_FILE)


#%% 

# We create lists of words for each question. 
# To train Word2Vec, it is better to keep the stop words
df['words1'] = df['question1'].apply(lambda x: question_to_words(str(x)))
df['words2'] = df['question2'].apply(lambda x: question_to_words(str(x)))


#%%

# Load the word2vec model
model = Word2Vec.load('w2v_300features_40minwords_20context.model')


#%%

# get a list of vectors for each question
def makeFeatureVec(words, model, num_features):
    ''' create a list of vectors from a list of words
    parameters
    ----------
    words    list
        the list of words that is transformed
    
    model    object
        a trained word2vec model
        
    return
    ------
    a list of vectors
    '''
    
    feature_vec = np.zeros((num_features,), dtype='float32')
    
    index2word_vec = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_vec:
            feature_vec = np.append(feature_vec, model[word])
            
    return feature_vec


#%%

num_features = 300

df['q1_vectors'] = df['words1'].apply(lambda x: makeFeatureVec(str(x), model, num_features))
df['q2_vectors'] = df['words2'].apply(lambda x: makeFeatureVec(str(x), model, num_features))
    