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
                   pipeline_arguments,
                   pipelines,
                   model_fit,
                   model_prediction,
                   model_evaluation,
                   get_model_attributions,
                   get_feature_targets
                   )
from sklearn.model_selection import LeaveOneGroupOut

model_name          = 'SVM' # change model name
experiment_type     = 'LOO'
target_attributes   = 'confidence' # change folder name
domain              = 'Perception' # change domain
reg_clf             = 'classification' # change type
split_data          = 'no-split'
data_dir            = '../data'
model_dir           = '../models/{}_{}_{}_{}'.format(*[model_name,experiment_type,target_attributes,split_data])
working_dir         = '../data/dataset/'
working_data        = glob(os.path.join(working_dir, f"{domain}.csv"))
working_df_name     = os.path.join(data_dir,target_attributes,f'{domain}.csv')
saving_dir          = f'../results/{reg_clf}/{target_attributes}/{experiment_type}'
batch_size          = 32
n_features          = 7 if target_attributes != 'confidence-accuracy' else 14
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
debug               = True
df_def              = pd.read_csv(working_df_name,)
unique_filenames    = pd.unique(df_def['filename'])
idx                 = 3 # change index
is_rnn              = True if model_name == 'RNN' else False
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
                                                            normalize_targets   = False,
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
                               n_sample         = [],
                               source           = [],
                               sub_name         = [],
                               best_params      = [],
                               feature_type     = [],
                               source_data      = [],
                               target_data      = [],
                               special          = [],
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
        X_,y_,g_train   = features[train_],targets[train_],groups[train_]
        X_test, y_test  = features[test]  ,targets[test]
        acc_test        = accuracies[test]
        acc_train_      = accuracies[train_]
        
        _train,_test    = [],[]
        for temp in cv.split(X_,y_,groups = g_train):
            _train.append(temp[0])
            _test.append(temp[1])
        np.random.seed(12345)
        if len(_train) > 100:
            _idx = np.random.choice(len(_train),size = 100,replace = False)
            _train = [_train[item] for item in _idx]
        # make the model
        xargs = pipeline_arguments()
        pipeline = pipelines(xargs)[f'{model_name.lower()}_{reg_clf}']
        # fit the model
        pipeline = model_fit(pipeline,
                             cv         = zip(_train,_test),
                             X_train    = X_,
                             y_train    = y_,
                             model_name = model_name.lower(),
                             reg_clf    = reg_clf,
                             n_features = n_features,
                             )
        
        # test the model
        y_pred      = model_prediction(pipeline,X_test,reg_clf = reg_clf,is_rnn = is_rnn,)
        # evaludate the model
        CR_dim      = y_pred.shape[1] if reg_clf == 'classification' else 4
        scores      = model_evaluation(y_test,y_pred,
                                       confidence_range = CR_dim,
                                       reg_clf          = reg_clf,
                                       is_rnn           = is_rnn,
                                       )
        # get the weights
        properties  = get_model_attributions(pipeline,X_test,y_test,
                                             model_name = model_name.lower(),
                                             reg_clf    = reg_clf,
                                             )
        # get parameters
        params      = pipeline.best_estimator_.get_params()
        
        # save the results
        if model_name == 'SVM' and reg_clf == 'classification':
            for ii_row,row in enumerate(properties):
                results['fold'                          ].append(fold)
                results['score'                         ].append(scores)
                results['n_sample'                      ].append(X_test.shape[0])
                results['source'                        ].append('same')
                results['sub_name'                      ].append(np.unique(groups[test])[0])
                [results[f'features T-{n_features - ii_item}'].append(item) for ii_item,item in enumerate(row)]
                results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
                results['feature_type'                  ].append(target_attributes)
                results['source_data'                   ].append(domain)
                results['target_data'                   ].append(domain)
                results['special'                       ].append(ii_row + 1)
        else:
            results['fold'                          ].append(fold)
            results['score'                         ].append(scores)
            results['n_sample'                      ].append(X_test.shape[0])
            results['source'                        ].append('same')
            results['sub_name'                      ].append(np.unique(groups[test])[0])
            [results[f'features T-{n_features - ii}'].append(item) for ii,item in enumerate(properties)]
            results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
            results['feature_type'                  ].append(target_attributes)
            results['source_data'                   ].append(domain)
            results['target_data'                   ].append(domain)
            results['special'                       ].append('None')
        
        results_to_save = pd.DataFrame(results)
    else:
        results_to_save = pd.DataFrame(results)
        
    results_to_save.to_csv(csv_name,index = False)