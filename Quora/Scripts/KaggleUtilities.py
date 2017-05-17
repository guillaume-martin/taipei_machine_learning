#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 08:41:18 2017

@author: guillaume
"""

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class TextProcessing(object):
    
    
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        ''' transform a text into a list of lowercase words
        parameters
        ----------
        text    string
            the text to be tokenized
        
        remove_stopwords    boolean
            True if the stop words need to be removed. False otherwise
            
        stem_words    boolean
            True if words need to be reduced to stem. False otherwise
            
        return
        ------
        a list of words
        '''
        
        # convert to lowercase
        text = text.lower()
        
        # remove punctuation and split
        text = re.findall(r'\w+', text, flags = re.UNICODE)
        
        # remove stop words
        words=[]
        if remove_stopwords:
            stops = stopwords.words('english')
            words = [word for word in text if not word in stops]
                    
        # stem words
        if stem_words:
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(word) for word in words]
            
        if len(words) > 0:
            return words
        else:
            return text