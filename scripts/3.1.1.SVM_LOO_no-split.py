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

from utils import (check_column_type,
                   build_SVMRegressor,
                   get_feature_targets
                   )
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import explained_variance_score,r2_score

model_name          = 'SVM'
experiment_type     = 'LOO'
target_attributes   = 'confidence' # change folder name
domain              = 'Perception' # change domain
split_data          = 'no-split'
data_dir            = '../data'
model_dir           = '../models/{}_{}_{}_{}'.format(*[model_name,experiment_type,target_attributes,split_data])
working_dir         = '../data/dataset/'
working_data        = glob(os.path.join(working_dir, f"{domain}.csv"))
working_df_name     = os.path.join(data_dir,target_attributes,f'{domain}.csv')
saving_dir          = f'../results/{target_attributes}/{experiment_type}'
batch_size          = 32
n_features          = 7 if target_attributes != 'confidence-accuracy' else 14
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
debug               = True
df_def              = pd.read_csv(working_df_name,)
unique_filenames    = pd.unique(df_def['filename'])
idx                 = 0 # change index
# pick one of the csv files
filename            = unique_filenames[idx]
df_sub              = df_def[df_def['filename'] == filename]
df_sub              = check_column_type(df_sub)

features, targets, groups, accuracies = get_feature_targets(df_sub,
                                                            n_features          = n_features,
                                                            time_steps          = time_steps,
                                                            target_attributes   = target_attributes,
                                                            group_col           = 'sub',
                                                            normalize_features  = False,
                                                            normalize_targets   = True,
                                                            )

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
                               source_data      = [],
                               target_data      = [],
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
        acc_test        = accuracies[test]
        acc_train_      = accuracies[train_]
        
        # make the model
        pipeline = make_pipeline(StandardScaler(),
                                 build_SVMRegressor())
        
        model = GridSearchCV(pipeline,
                             {'linearsvr__C':np.logspace(0,5,6),
                              'linearsvr__loss':['epsilon_insensitive', # L1 loss
                                                 'squared_epsilon_insensitive',# L2 loss
                                                 ]},
                             scoring    = 'explained_variance',
                             n_jobs     = -1,
                             cv         = 10,
                             verbose    = 1,
                             )
        # train the model and validate the model
        model.fit(X_,y_)
        gc.collect()
        # test the model
        y_pred      = model.predict(X_test)
        scores      = explained_variance_score(y_test,y_pred,)
        
        # get the weights
        properties  = model.best_estimator_.steps[-1][-1].coef_
        # get parameters
        params      = model.best_estimator_.get_params()
        
        # save the results
        results['fold'                          ].append(fold)
        results['score'                         ].append(scores)
        results['r2'                            ].append(r2_score(y_test,y_pred,))
        results['n_sample'                      ].append(X_test.shape[0])
        results['source'                        ].append('same')
        results['sub_name'                      ].append(np.unique(groups[test])[0])
        [results[f'features T-{n_features - ii}'].append(item) for ii,item in enumerate(properties)]
        results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
        results['feature_type'                  ].append(target_attributes)
        results['source_data'                   ].append(domain)
        results['target_data'                   ].append(domain)
        
        results_to_save = pd.DataFrame(results)
    else:
        results_to_save = pd.DataFrame(results)
        
    results_to_save.to_csv(csv_name,index = False)