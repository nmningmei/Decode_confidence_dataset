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

from utils import check_column_type,scoring_func,get_properties

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils           import shuffle as util_shuffle
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline

experiment          = ['_adequacy','LOO','SVM']
data_dir            = '../data'
model_dir           = f'../models/{experiment[1]}_{experiment[2]}'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],'all_data.csv')
saving_dir          = f'../results/{experiment[0]}/{experiment[1]}'
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
property_name       = 'weight'
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
csv_name    = os.path.join(saving_dir,f'{experiment[2]} cross validation results LOO ({kk}).csv')
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
    for jj in range(confidence_range):
        results[f'{property_name} T-{time_steps - ii} C-{jj}'] = []

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
        
        svc = LinearSVC(class_weight='balanced',random_state = 12345)
        model = CalibratedClassifierCV(svc,cv = 5)
        model = make_pipeline(StandardScaler(),model)
        model.fit(X_[_idx],y_[_idx])
        preds_test  = model.predict_proba(X_test)
        print(scoring_func(to_categorical(y_[_idx] - 1, num_classes = confidence_range), model.predict_proba(X_[_idx]),confidence_range=confidence_range,need_normalize=True))
        print(f'get {property_name}')
        properties = get_properties(model,experiment[-1])
        
        print('on test')
        for acc_trial_test in [0,1]:
            _idx_test, = np.where(acc_test == acc_trial_test)
            if len(_idx_test) > 1:
                score_test = scoring_func(y_test[_idx_test],preds_test[_idx_test],
                                          confidence_range = confidence_range,
                                          need_normalize=True)
                
                print(score_test)
                results['fold'].append(fold)
                results['score'].append(np.mean(score_test))
                results['n_sample'].append(X_test[_idx_test].shape[0])
                results['source'].append('same')
                results['sub_name'].append(np.unique(groups[test])[0])
                results['accuracy_train'].append(acc_trial_train)
                results['accuracy_test'].append(acc_trial_test)
                [results[f'{property_name} T-{time_steps - ii} C-{jj}'].append(
                        properties[jj,ii]
                        ) for ii in range(time_steps) for jj in range(confidence_range)]
        gc.collect()
        
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_name,index = False)

