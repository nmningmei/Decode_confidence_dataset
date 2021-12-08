#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:09:46 2021

@author: nmei
"""
import os,gc

from glob import glob

import pandas as pd
import numpy as np

from utils import check_column_type,build_RF
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import explained_variance_score,r2_score
from sklearn.inspection import permutation_importance

model_name          = 'RF'
experiment_type     = 'LOO'
target_attributes   = 'confidence' # change folder name
split_data          = 'no-split'
data_dir            = '../data'
model_dir           = '../models/{}_{}_{}_{}'.format(*[model_name,experiment_type,target_attributes,split_data])
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,target_attributes,experiment_type,'all_data.csv')
saving_dir          = f'../results/{target_attributes}/{experiment_type}'
batch_size          = 32
n_features          = 7 if target_attributes != 'confidence-accuracy' else 14
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
debug               = True

df_def          = pd.read_csv(working_df_name,)

# pick one of the csv files
filename = '../data/4-point/data_Bang_2019_Exp1.csv' # change file name
df_sub = df_def[df_def['filename'] == filename]
df_sub = check_column_type(df_sub)

if target_attributes == 'confidence-accuracy':
    features= df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values / np.concatenate([[4]*time_steps,[1]*time_steps])
elif target_attributes == 'confidence':
    features= df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values / 4 # scale the features
else:
    features= df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values
targets     = df_sub["targets"].values / 4 # scale the targets
groups      = df_sub["sub"].values
accuraies   = df_sub['accuracy'].values
kk          = filename.split('/')[-1].split(".csv")[0]
cv          = LeaveOneGroupOut()
csv_name    = os.path.join(saving_dir,f'results_{kk}_{model_name}_{target_attributes}.csv')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
print(csv_name)

if not os.path.exists(csv_name):
    results             = dict(
                               fold             = [],
                               score            = [],
                               r2               = [],
                               n_sample         = [],
                               source           = [],
                               sub_name         = [],
                               best_params      = [],
                               feature_type     = [],
                               )
    for ii in range(n_features):
        results[f'features T-{n_features - ii}'] = []
else:
    temp = pd.read_csv(csv_name)
    results = {}
    for col_name in temp.columns:
        results[col_name] = list(temp[col_name].values)
print(cv.get_n_splits(features,targets,groups=groups))
for fold,(train_,test) in enumerate(cv.split(features,targets,groups=groups)):
    print(f'fold {fold}')
    if fold not in results['fold']:
        # leave out test data
        X_,y_           = features[train_],targets[train_]
        X_test, y_test  = features[test]  ,targets[test]
        acc_test        = accuraies[test]
        acc_train_      = accuraies[train_]
        
        # make the model
        model = GridSearchCV(build_RF(bootstrap = True,
                                      oob_score = False,),
                            {'n_estimators':np.logspace(0,3,4).astype(int),
                              'max_depth':np.arange(n_features) + 1},
                             scoring    = 'explained_variance',
                             n_jobs     = -1,
                             cv         = 10,
                             verbose    = 1,
                             )
        # train the model and validate the model
        model.fit(X_,y_)
        gc.collect()
        # test the model
        y_pred = model.predict(X_test)
        scores = explained_variance_score(y_test,y_pred,)
        
        # get the weights
        properties = permutation_importance(model.best_estimator_,
                                            X_test,
                                            y_test,
                                            scoring = 'explained_variance',
                                            n_repeats = 5,
                                            n_jobs = -1,
                                            random_state = 12345)
        # get parameters
        params = model.best_estimator_.get_params()
        
        # save the results
        results['fold'].append(fold)
        results['score'].append(scores)
        results['r2'].append(r2_score(y_test,y_pred,))
        results['n_sample'].append(X_test.shape[0])
        results['source'].append('same')
        results['sub_name'].append(np.unique(groups[test])[0])
        [results[f'features T-{n_features - ii}'].append(item) for ii,item in enumerate(properties['importances_mean'])]
        results['best_params'].append('|'.join(f'{key}:{value}' for key,value in params.items()))
        results['feature_type'].append(target_attributes)
    
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(csv_name,index = False)