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

from utils import check_column_type,build_RNN
from sklearn.model_selection import LeaveOneGroupOut,StratifiedShuffleSplit
from sklearn.metrics import explained_variance_score,r2_score
from tensorflow.keras import Model

model_name          = 'RNN'
experiment_type     = 'LOO'
target_attributes   = 'confidence-accuracy' # change folder name
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

df_def              = pd.read_csv(working_df_name,)

# pick one of the csv files
filename            = '../data/4-point/data_Bang_2019_Exp1.csv' # change file name
df_sub              = df_def[df_def['filename'] == filename]
df_sub              = check_column_type(df_sub)

if target_attributes == 'confidence-accuracy':
    features= df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values / np.concatenate([[4]*time_steps,[1]*time_steps])
else:
    features= df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values / 4 # scale the features
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
        # reshape for RNN
        if target_attributes == 'confidence-accuracy':
            features    = np.swapaxes(features.reshape(features.shape[0],2,time_steps),1,2)
            input_dim   = 2
        else:
            features    = features.reshape(features.shape[0],features.shape[-1],1)
            input_dim   = 1
        # leave out test data
        X_,y_           = features[train_],targets[train_]
        X_test, y_test  = features[test]  ,targets[test]
        acc_test        = accuraies[test]
        acc_train_      = accuraies[train_]
        # the for-loop does not mean any thing, we only take the last step/output of the for-loop
        for train,valid in StratifiedShuffleSplit(test_size = 0.2,
                                                  random_state = 12345).split(features[train_],
                                                                              targets[train_],
                                                                              groups=groups[train_]):
            X_train,y_train = X_[train],y_[train]
            X_valid,y_valid = X_[valid],y_[valid]
        # make the model
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model,callbacks = build_RNN(time_steps          = time_steps,
                                    confidence_range    = confidence_range,
                                    input_dim           = input_dim,
                                    model_name          = os.path.join(model_dir,
                                                                       f'{target_attributes}_{kk}_{fold}.h5'))
        # build hidden layer model
        hidden_model = Model(model.input,model.layers[1].output)
        # train the model and validate the model
        model.fit(X_train,
                  y_train,
                  batch_size        = batch_size,
                  epochs            = 1000,
                  validation_data   = (X_valid,y_valid),
                  shuffle           = True,
                  callbacks         = callbacks,
                  verbose           = debug,)
        gc.collect()
        # test the model
        y_pred = model.predict(X_test).flatten()
        scores = explained_variance_score(y_test,y_pred,)
        
        # get the weights
        properties = hidden_model.predict(X_test)[0].mean(0).flatten()
        # get parameters
        params = f'input(,{time_steps},{input_dim})->lstm(,{time_steps},1)->output(,1)'
        
        # save the results
        results['fold'].append(fold)
        results['score'].append(scores)
        results['r2'].append(r2_score(y_test,y_pred,))
        results['n_sample'].append(X_test.shape[0])
        results['source'].append('same')
        results['sub_name'].append(np.unique(groups[test])[0])
        [results[f'features T-{time_steps - ii}'].append(item) for ii,item in enumerate(properties)]
        results['best_params'].append(params)
        results['feature_type'].append(target_attributes)
    
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(csv_name,index = False)