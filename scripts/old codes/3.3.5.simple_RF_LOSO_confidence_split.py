#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:35:17 2019

@author: nmei
"""
import os
import gc
gc.collect() # clean garbage memory
from glob import glob

from tensorflow.keras.utils import to_categorical

import numpy  as np
import pandas as pd

from utils import check_column_type,scoring_func,build_RF

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils           import shuffle as util_shuffle
from sklearn.inspection      import permutation_importance
from sklearn.metrics         import make_scorer

experiment          = ['confidence','LOO','RF','past']
data_dir            = '../data'
model_dir           = f'../models/{experiment[1]}_{experiment[2]}'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],f'all_data_{experiment[3]}.csv')
saving_dir          = f'../results/{experiment[0]}/{experiment[1]}'
batch_size          = 32
time_steps          = 3 # change here
confidence_range    = 4
n_jobs              = -1
verbose             = 1
property_name       = 'feature_importance'
debug               = True

df_def          = pd.read_csv(working_df_name,)


# pick one of the csv files
filename = '../data/4-point/data_Bang_2019_Exp1.csv'
df_sub = df_def[df_def['filename'] == filename]
df_sub = check_column_type(df_sub)

#for (filename),df_sub in df_def.groupby(["filename"]):
features    = df_sub[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets     = df_sub["targets"].values.astype(int)
groups      = df_sub["sub"].values
accuraies   = df_sub['accuracy'].values
kk          = filename.split('/')[-1].split(".csv")[0]
cv          = LeaveOneGroupOut()
csv_name    = os.path.join(saving_dir,f'{experiment[2]} {experiment[3]} cross validation results LOO ({kk}).csv')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
print(csv_name)

results             = dict(
                           fold             = [],
                           score            = [],
                           n_sample         = [],
                           source           = [],
                           sub_name         = [],
                           accuracy_train   = [],
                           accuracy_test    = [],
                           )
for ii in range(time_steps):
    results[f'{property_name} T-{time_steps - ii}'] = []

for fold,(train_,test) in enumerate(cv.split(features,targets,groups=groups)):
    # leave out test data
    X_,y_           = features[train_],targets[train_]
    X_test, y_test  = features[test]  ,targets[test]
    acc_test        = accuraies[test]
    acc_train_      = accuraies[train_]
    
    # make sure processing the X_test and y_test only once
    # X_test  = to_categorical(X_test  - 1, num_classes = confidence_range).reshape(-1,time_steps*confidence_range)
    y_test  = to_categorical(y_test  - 1, num_classes = confidence_range)
    
    # split into train and validation data
    np.random.seed(12345)
    X_,y_ = util_shuffle(X_,y_)
    for acc_trial_train in [0,1]:
        _idx, = np.where(acc_train_ == acc_trial_train)
        
        model = build_RF(n_estimators = 500,n_jobs = -1)
        model.fit(X_[_idx],y_[_idx])
        preds_test  = model.predict_proba(X_test)
        
        print('on test')
        for acc_trial_test in [0,1]:
            _idx_test, = np.where(acc_test == acc_trial_test)
            if len(_idx_test) > 1:
                score_test = scoring_func(y_test[_idx_test],preds_test[_idx_test],
                                          confidence_range = confidence_range,
                                          need_normalize=True)
                scorer = make_scorer(scoring_func,needs_proba=True,
                                     **{'confidence_range':confidence_range,
                                        'need_normalize':True,
                                        'one_hot_y_true':False})
                _feature_importance = permutation_importance(model,
                                                             X_test[_idx_test],
                                                             y_test[_idx_test],
                                                             scoring         = scorer,
                                                             n_repeats       = 10,
                                                             n_jobs          = -1,
                                                             random_state    = 12345,
                                                             )
                feature_importance = _feature_importance['importances_mean']
                print(score_test)
                results['fold'].append(fold)
                results['score'].append(np.mean(score_test))
                results['n_sample'].append(X_test[_idx_test].shape[0])
                results['source'].append('same')
                results['sub_name'].append(np.unique(groups[test])[0])
                results['accuracy_train'].append(acc_trial_train)
                results['accuracy_test'].append(acc_trial_test)
                [results[f'{property_name} T-{time_steps - ii}'].append(item) for ii,item in enumerate(feature_importance)]
        gc.collect()
        
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_name,index = False)

