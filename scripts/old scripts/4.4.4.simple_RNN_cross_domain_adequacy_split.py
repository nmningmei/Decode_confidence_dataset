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

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

import numpy  as np
import pandas as pd

from utils import build_RNN,scoring_func

from sklearn.model_selection import StratifiedShuffleSplit,LeaveOneGroupOut

experiment          = ['adequacy','cross_domain','RNN','past']
property_name       = 'hidden states' # or hidden states or weight
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
    results[f'{property_name} T-{time_steps - ii}'] = []

features    = df_source[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets     = df_source["targets"].values.astype(int)
groups      = df_source["filename"].values
accuracies  = df_source['accuracy'].values
sec_groups  = df_source['sub'].values

csv_saving_name     = os.path.join(result_dir,f'{experiment[2]}_{experiment[0]}_{experiment[3]} results.csv')
cv = LeaveOneGroupOut()
for fold,(_,_train) in enumerate(cv.split(features,targets,groups = groups)):
    X_,Y_,Z_ = features[_train],targets[_train],sec_groups[_train]
    # the for-loop does not mean any thing, we only take the last step/output of the for-loop
    for train,valid in StratifiedShuffleSplit(test_size = 0.2,random_state = 12345).split(X_,Y_,Z_):
        X_train,y_train,group_train,acc_train = X_[train],Y_[train],Z_[train],accuracies[train]
        X_valid,y_valid,gruop_valid,acc_valid = X_[valid],Y_[valid],Z_[valid],accuracies[valid]
    
    X_train = to_categorical(X_train - 1, num_classes = confidence_range)
    X_valid = to_categorical(X_valid - 1, num_classes = confidence_range)
    
    y_train = to_categorical(y_train - 1, num_classes = confidence_range)
    y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)
    
    model_name     = os.path.join(model_dir,f'{"_".join(experiment)}.h5')
    model,callbacks = build_RNN(time_steps,confidence_range,model_name)
    
    model.fit(X_train,
              y_train,
              batch_size        = batch_size,
              epochs            = 1000,
              validation_data   = (X_valid,y_valid),
              shuffle           = True,
              callbacks         = callbacks,
              verbose           = 1,
              )
    del model
    model = tf.keras.models.load_model(model_name)
    # freeze the model
    for layer in model.layers:
        layer.trainable = False
    hidden_model = Model(model.input,model.layers[1].output)
    
    # test phase
    for (filename,sub_name,target_domain),df_sub in df_target.groupby(['filename','sub','domain']):
        df_sub
        features_        = df_sub[[f"feature{ii + 1}" for ii in range(time_steps)]].values
        targets_         = df_sub["targets"].values.astype(int)
        groups_          = df_sub["sub"].values
        X_test,y_test    = features_.copy(),targets_.copy()
        acc_test         = df_sub['accuracy'].values
        X_test           = to_categorical(X_test - 1, num_classes = confidence_range)
        y_test           = to_categorical(y_test - 1, num_classes = confidence_range)
        
        preds_test  = model.predict(X_test.astype('float32'), batch_size=batch_size)
        print(f'get {property_name}')
        hidden_state_test,h_state_test,c_state_test = hidden_model.predict(X_test,
                                                                            batch_size = batch_size,
                                                                            verbose = 1)
        score_test = scoring_func(y_test,preds_test,
                                  confidence_range = confidence_range)
        print(score_test)
        results['fold'].append(fold)
        results['score'].append(np.mean(score_test))
        results['n_sample'].append(X_test.shape[0])
        results['source'].append(target_domain)
        results['sub_name'].append(sub_name)
        # results['accuracy_train'].append(acc_trial_train)
        # results['accuracy_test'].append(acc_trial_test)
        results['filename'].append(filename)
        [results[f'{property_name} T-{time_steps - ii}'].append(hidden_state_test.mean(0)[ii,0]) for ii in range(time_steps)]
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_saving_name,index = False)









