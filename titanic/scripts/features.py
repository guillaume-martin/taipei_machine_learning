#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:18:11 2016

@author: guillaume
"""

import pandas as pd

#%%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#%%
def one_hot_encode(dataframe,columns, drop=True):
    ''' performs one-hot encoding on a set of columns of a dataframe        
    '''
    for col in columns:
        df_encoded = pd.get_dummies(dataframe[col], prefix=col)
        dataframe = pd.concat([dataframe,df_encoded], axis=1)
        if drop == True:
            dataframe.drop([col], inplace=True, axis=1)
   
    return dataframe

# reuse the get_title function from Calvin
def get_title(string):
    officers = set(['Capt', 'Col', 'Major', 'Rev'])
    royalties = set(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'])
    mr = set(['Mr', 'Mister'])
    mrs = set(['Mme', 'Mrs'])
    ms = set(['Mlle', 'Miss', 'Ms'])
    
    for title in officers:
        if string.find(title) != -1:
            return 'Officer'
    for title in royalties:
        if string.find(title) != -1:
            return 'Royalty'
    for title in ms:
        if string.find(title) != -1:
            return 'Miss'
    if string.find('Master') != -1:
            return 'Master'
    if string.find('Dr.') != -1:
            return 'Dr'
    for title in mrs:
        if string.find(title) != -1:
            return 'Mrs'
    for title in mr:
        if string.find(title) != -1:
            return 'Mr'    
            
#%%
def preprocess_data(dataframe):
    # get the titles
    dataframe['Title'] = dataframe['Name'].apply(get_title)
    
    # get family size
    dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch'] + 1

    dataframe['Fare'].loc[dataframe['Fare'].isnull()] = dataframe['Fare'].mean()
    
    # extract deck number
    dataframe['Deck'] = dataframe['Cabin'].astype(str).str[0]
    dataframe.drop('Cabin', inplace=True, axis=1)
    dataframe['Deck'][dataframe.Deck == 'n'] = 'U0'
   
    # fill missing embarked
    dataframe['Embarked'].fillna('S', inplace=True)

    # fill missing age
    titles = ['Officer', 'Royalty', 'Miss', 'Master', 'Dr', 'Mrs', 'Mr']
    for title in titles:
        med = dataframe.loc[dataframe['Title'] == title].loc[:, 'Age'].median()
        dataframe.loc[:, 'Age'].loc[(dataframe['Age'].isnull()) & (dataframe['Title'] == title)] = med 

    # create age bins
    age_bins = [0,15,25,35,45,55,65,75,85]
    dataframe['AgeBins'] = pd.cut(dataframe['Age'], age_bins)

    # one-hot encoding
    columns = ['Pclass','Sex','AgeBins','Embarked','Title','Deck']
    dataframe = one_hot_encode(dataframe, columns, True)

    # drop useless columns
    dataframe.drop(['Name','Ticket'], inplace=True, axis=1)
    
    return dataframe
    
#%%

    processed_train = preprocess_data(train)
    processed_test = preprocess_data(test)
    
#%%
    
    # export 
    processed_train.to_csv('../input/processed_train.csv')
    processed_test.to_csv('../input/processed_test.csv')
    