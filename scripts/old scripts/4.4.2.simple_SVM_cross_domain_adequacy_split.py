#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:10:54 2019

@author: nmei
"""

import os
import gc
gc.collect() # clean garbage memory
from glob import glob

from tensorflow.keras.utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut

import numpy  as np
import pandas as pd

from utils import scoring_func,get_properties

experiment          = ['adequacy','cross_domain','SVM','past']
property_name       = 'weight' # or hidden states or weight
data_dir            = '../data/'
model_dir           = os.path.join('../models',experiment[0],experiment[1],)
source_dir          = '../data/4-point'
target_dir          = '../data/targets/*/'
result_dir          = os.path.join('../results/',experiment[0],experiment[1],)
hidden_dir          = os.path.join('../results/',experiment[0],experiment[1],property_name)
source_df_name      = os.path.join(data_dir,experiment[0],experiment[1],f'source_{experiment[3]}.csv')
target_df_name      = os.path.join(data_dir,experiment[0],experiment[1],f'target_{experiment[3]}.csv')
batch_size          = 32
time_steps          = 3 # change here
confidence_range    = 4
n_jobs              = -1
split               = False # split the data into high and low dprime-metadrpime


for d in [model_dir,result_dir,hidden_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

df_source           = pd.read_csv(source_df_name)
df_target           = pd.read_csv(target_df_name)

df_target['domain'] = df_target['filename'].apply(lambda x:x.split('/')[3].split('-')[0])

results             = dict(fold             = [],
                           score            = [],
                           n_sample         = [],
                           source           = [],
                           sub_name         = [],
                           filename         = [],
                           # accuracy_train   = [],
                           # accuracy_test    = [],
                           )
for ii in range(time_steps):
    for jj in range(confidence_range):
        results[f'{property_name} T-{time_steps - ii} C-{jj}'] = []

features    = df_source[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets     = df_source["targets"].values.astype(int)
groups      = df_source["filename"].values#.apply(lambda x:x.split('/')[-1].split('.')[0]) + "_" + df_source['sub'].astype(str)
accuracies  = df_source['accuracy'].values

csv_saving_name     = os.path.join(result_dir,f'{experiment[2]}_{experiment[0]}_{experiment[3]} results.csv')
cv = LeaveOneGroupOut()
for fold,(_,train) in enumerate(cv.split(features,targets,groups = groups)):
    X_,Y_,Z_ = features[train],targets[train],groups[train]
    # from sklearn.linear_model import LogisticRegressionCV
    print('fitting ...')
    clf = LinearSVC(dual = False,class_weight = 'balanced',random_state = 12345)
    # clf = LogisticRegressionCV(Cs = np.logspace(-3,3,7),class_weight = 'balanced',random_state = 12345,n_jobs = -1,)
    model = CalibratedClassifierCV(clf,cv = 5)
    model = make_pipeline(StandardScaler(),
                          model)
    model.fit(X_,Y_)
    
    print(f'get {property_name}')
    properties = get_properties(model,experiment[2])
    
    
    # test phase
    for (filename,sub_name,target_domain),df_sub in df_target.groupby(['filename','sub','domain']):
        df_sub
        features_        = df_sub[[f"feature{ii + 1}" for ii in range(time_steps)]].values
        targets_         = df_sub["targets"].values.astype(int)
        X_test,y_test    = features_.copy(),targets_.copy()
        acc_test         = df_sub['accuracy'].values
        # X_test           = to_categorical(X_test - 1, num_classes = confidence_range).reshape(-1,time_steps*confidence_range)
        y_test           = to_categorical(y_test - 1, num_classes = confidence_range)
        
        preds_test  = model.predict_proba(X_test)
        
        
        score_test = scoring_func(y_test,preds_test,
                                  confidence_range = confidence_range,
                                  need_normalize = True,)
        print(score_test)
        results['fold'].append(fold)
        results['score'].append(np.mean(score_test))
        results['n_sample'].append(X_test.shape[0])
        results['source'].append(target_domain)
        results['sub_name'].append(sub_name)
        # results['accuracy_train'].append(acc_trial_train)
        # results['accuracy_test'].append(acc_trial_test)
        results['filename'].append(filename)
        [results[f'{property_name} T-{time_steps - ii} C-{jj}'].append(
            properties[jj,ii]
            ) for ii in range(time_steps) for jj in range(confidence_range)]
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_saving_name,index = False)







