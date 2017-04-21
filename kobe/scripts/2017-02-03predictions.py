#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:05:51 2017

@author: guillaume
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

#%%

df = pd.read_pickle('../input/processed_data.pickle')

# remove game_date
df.drop(['game_date'], axis=1, inplace=True)

#%% 

best_features = ['MA10', 'MA20', 'MA50', 'MA100', 'action_type_Alley Oop Dunk Shot',
       'action_type_Cutting Layup Shot', 'action_type_Driving Bank shot',
       'action_type_Driving Dunk Shot',
       'action_type_Driving Finger Roll Layup Shot',
       'action_type_Driving Finger Roll Shot',
       'action_type_Driving Floating Bank Jump Shot',
       'action_type_Driving Floating Jump Shot',
       'action_type_Driving Jump shot', 'action_type_Driving Layup Shot',
       'action_type_Driving Reverse Layup Shot',
       'action_type_Driving Slam Dunk Shot', 'action_type_Dunk Shot',
       'action_type_Fadeaway Bank shot', 'action_type_Fadeaway Jump Shot',
       'action_type_Finger Roll Layup Shot',
       'action_type_Finger Roll Shot', 'action_type_Floating Jump shot',
       'action_type_Hook Bank Shot', 'action_type_Hook Shot',
       'action_type_Jump Bank Shot', 'action_type_Jump Hook Shot',
       'action_type_Jump Shot', 'action_type_Layup Shot',
       'action_type_Pullup Bank shot', 'action_type_Pullup Jump shot',
       'action_type_Putback Dunk Shot',
       'action_type_Putback Slam Dunk Shot',
       'action_type_Reverse Layup Shot',
       'action_type_Reverse Slam Dunk Shot',
       'action_type_Running Bank shot',
       'action_type_Running Finger Roll Layup Shot',
       'action_type_Running Finger Roll Shot',
       'action_type_Running Hook Shot', 'action_type_Running Jump Shot',
       'action_type_Running Reverse Layup Shot',
       'action_type_Running Tip Shot', 'action_type_Slam Dunk Shot',
       'action_type_Step Back Jump shot', 'action_type_Tip Shot',
       'action_type_Turnaround Fadeaway shot',
       'action_type_Turnaround Finger Roll Shot',
       'action_type_Turnaround Hook Shot',
       'action_type_Turnaround Jump Shot', 'combined_shot_type_Bank Shot',
       'combined_shot_type_Dunk', 'combined_shot_type_Hook Shot',
       'combined_shot_type_Jump Shot', 'combined_shot_type_Layup',
       'combined_shot_type_Tip Shot', 'period_2', 'period_3', 'period_4',
       'period_7', 'season_2015-16', 'shot_type_2PT Field Goal',
       'shot_type_3PT Field Goal', 'shot_zone_area_Back Court(BC)',
       'shot_zone_area_Center(C)', 'shot_zone_area_Left Side Center(LC)',
       'shot_zone_area_Right Side Center(RC)',
       'shot_zone_basic_Above the Break 3', 'shot_zone_basic_Backcourt',
       'shot_zone_basic_Left Corner 3', 'shot_zone_basic_Restricted Area',
       'shot_zone_range_Back Court Shot',
       'shot_zone_range_Less Than 8 ft.', 'year_2013', 'year_2015',
       'opponent_BKN', 'opponent_GSW', 'opponent_MIL', 'opponent_NJN',
       'opponent_OKC', 'opponent_SEA', 'opponent_TOR', 'venue_away',
       'venue_home']

#%%

# We need an index to filter examples prior to the prediction.
# At this point, the dataframe is already sorted chronologically.
# We just need to reset the index
df.reset_index(inplace=True)

#%%

# There are some NaN in moving averages features.
# we replace them with 0.4 (the average success rate of kobe)
columns = ['MA10','MA20','MA50','MA100','MA200']

for col in columns:
    df[col].fillna(0.4, inplace=True)

#%%

# We now split the dataframe into a training set and a test set
df_train = df.dropna(subset=['shot_made_flag'])
df_test = df[df['shot_made_flag'].isnull()]

# We remove the shot_made_flag from the test set
# All values are NaN
df_test.drop(['shot_made_flag'],axis=1,inplace=True)
             
#%%

'''
    ada = AdaBoostClassifier()
    
    for example in df_test:
        i = example's index
        train_subset = df_train where index < i
        X = np.array(train_subset.drop('shot_made_flag'))
        y = np.array(train_subset['shot_made_flag'])
        ada.fit(X,y)
        ada.proba_predict(example)
        save prediction with shot id
'''
import csv
import warnings; warnings.simplefilter('ignore')

ada = AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1000, max_iter=100, multi_class='ovr',
          n_jobs=1, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False),
          learning_rate=0.001, n_estimators=10, random_state=None)

i = 1
for index, row in df_test.iterrows():
        
    shot_id = df_test.get_value(index,'shot_id')
    
    print('Prediction # ' + str(i) + ', index = ' + str(index) + ', shot_id = ' + str(shot_id))
    i += 1
    
    # create a subset to train the classifier
    train_subset = df_train[df_train.index < index]
    #X = np.array(train_subset.drop(['shot_id','shot_made_flag'], axis=1))
    X = np.array(train_subset[best_features])
    y = np.array(train_subset['shot_made_flag'])
    ada.fit(X,y)
    
    # make a prediction for row
    X_test = np.array(df_test[best_features].loc[index])
    y_sub = ada.predict_proba(X_test)    
    
    # save the prediction to a csv file    
    file = '../submissions/lr_ada_20170202_01.csv'
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([shot_id,y_sub[0][0],y_sub[0][1]])
    