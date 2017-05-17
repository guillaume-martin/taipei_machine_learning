#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 08:25:45 2017

@author: guillaume
"""

# Import packages
import numpy as np
import csv, datetime, time, json
from KaggleUtilities import TextProcessing

#%%
# Setting variables
DATA_DIR = '../Data/'
TRAINING_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
WORD2VECMODEL = 'w2v_300features_40minwords_20context.model'
 

#%%
# Load the data
question1 = []
question2 = []
is_duplicate = []

with open(DATA_DIR + TRAINING_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter = ',')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
        is_duplicate.append(row['is_duplicate'])
   
     
#%%
# We tokenize the questions
words1 = []
words2 = []
for sentence in question1:
    words = TextProcessing.text_to_wordlist(sentence)
    words1.append(words)
    
for sentence in question2:
    words = TextProcessing.text_to_wordlist(sentence)
    words2.append(words)
