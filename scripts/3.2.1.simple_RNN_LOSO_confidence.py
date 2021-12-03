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

import tensorflow as tf
from tensorflow.keras       import layers, Model, optimizers, losses
from tensorflow.keras.utils import to_categorical

import numpy  as np
import pandas as pd

from utils import make_CallBackList,check_column_type,scoring_func

from sklearn.model_selection import LeaveOneGroupOut,StratifiedShuffleSplit
from sklearn.utils           import shuffle as util_shuffle

experiment          = ['confidence','LOO','RNN']
data_dir            = '../data'
model_dir           = f'../models/{experiment[0]}/{experiment[1]}_{experiment[2]}'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],'all_data.csv')
saving_dir          = f'../results/{experiment[0]}/{experiment[1]}'
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
property_name       = 'hidden_state'
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

if not os.path.exists(csv_name):
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
    get_folds = []
else:
    results = pd.read_csv(csv_name)
    get_folds = pd.unique(results['fold'])
    results = {col_name:list(results[col_name].values) for col_name in results.columns}

for fold,(train_,test) in enumerate(cv.split(features,targets,groups=groups)):
    model_name  = os.path.join(model_dir,f'LOO_{kk}_fold{fold + 1}.h5')
    print(model_name)
    if fold not in get_folds:
        # leave out test data
        X_,y_           = features[train_],targets[train_]
        X_test, y_test  = features[test]  ,targets[test]
        acc_test        = accuraies[test]
        acc_train_      = accuraies[train_]
        
        # make sure processing the X_test and y_test only once
        X_test  = to_categorical(X_test  - 1, num_classes = confidence_range)#.reshape(-1,time_steps*confidence_range)
        y_test  = to_categorical(y_test  - 1, num_classes = confidence_range)
        
        # split into train and validation data
        np.random.seed(12345)
        X_,y_ = util_shuffle(X_,y_)
        for acc_trial_train in [0,1]:
            _idx, = np.where(acc_train_ == acc_trial_train)
            
            # the for-loop does not mean any thing, we only take the last step/output of the for-loop
            for train,valid in StratifiedShuffleSplit(test_size = 0.2,
                                                      random_state = 12345).split(features[train_][_idx],
                                                                                  targets[train_][_idx],
                                                                                  groups=groups[train_][_idx]):
                X_train,y_train = X_[_idx][train],y_[_idx][train]
                X_valid,y_valid = X_[_idx][valid],y_[_idx][valid]
                acc_valid = acc_train_[_idx][valid]
           
            # reset the GPU memory
            tf.keras.backend.clear_session()
            try:
                tf.random.set_random_seed(12345) # tf 1.0
            except:
                tf.random.set_seed(12345) # tf 2.0
                
            # build a 3-layer RNN model
            inputs                  = layers.Input(shape     = (time_steps,4),# time steps by features 
                                                   name      = 'inputs')
            # the recurrent layer
            lstm,state_h,state_c    = layers.LSTM(units             = 1,
                                                  return_sequences  = True,
                                                  return_state      = True,
                                                  name              = "lstm")(inputs)
            # from the LSTM layer, we will have an output with time steps by features, but 
            dimension_squeeze       = layers.Lambda(lambda x:tf.keras.backend.squeeze(x,2),
                                                    name            = 'squeeze')(lstm)
            outputs                 = layers.Dense(4,
                                                   name             = "output",
                                                   activation       = "softmax")(dimension_squeeze)
            model                   = Model(inputs,
                                            outputs)
            
            model.compile(optimizer     = optimizers.SGD(lr = 1e-2),
                          loss          = losses.binary_crossentropy,
                          metrics       = ['mse'])
            # early stopping
            callbacks = make_CallBackList(model_name    = model_name,
                                          monitor       = 'val_loss',
                                          mode          = 'min',
                                          verbose       = 0,
                                          min_delta     = 1e-4,
                                          patience      = 5,
                                          frequency     = 1,)
            
            X_train = to_categorical(X_train - 1, num_classes = confidence_range)#.reshape(-1,time_steps*confidence_range)
            X_valid = to_categorical(X_valid - 1, num_classes = confidence_range)#.reshape(-1,time_steps*confidence_range)
            
            print('build hidden state model')
            hidden_model = Model(model.input,model.layers[1].output)
            
            y_train = to_categorical(y_train - 1, num_classes = confidence_range)
            y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)
            
            if not os.path.exists(os.path.join(*model_name.split('/')[:-1])):
                os.makedirs(os.path.join(*model_name.split('/')[:-1]))
            
            print('trained model not found, start training ...')
            model.fit(X_train,
                      y_train,
                      batch_size        = batch_size,
                      epochs            = 1000,
                      validation_data   = (X_valid,y_valid),
                      shuffle           = True,
                      callbacks         = callbacks,
                      verbose           = debug,
                      )
            
            del model
            model = tf.keras.models.load_model(model_name)
            # freeze the model
            for layer in model.layers:
                layers.trainable = False
            
            preds_valid = model.predict(X_valid.astype('float32'),batch_size=batch_size)
            preds_test  = model.predict(X_test.astype('float32'), batch_size=batch_size)
            
            print(f'get {property_name}')
            hidden_state_valid,h_state_valid,c_state_valid = hidden_model.predict(X_valid,
                                                                      batch_size = batch_size,
                                                                      verbose = 1)
            
            print('on train')
            temp_idx = acc_valid == acc_trial_train
            score_valid = scoring_func(y_valid[temp_idx],preds_valid[temp_idx],confidence_range = confidence_range)
            results['fold'].append(fold)
            results['score'].append(score_valid)
            results['n_sample'].append(X_valid[temp_idx].shape[0])
            results['source'].append('train')
            results['sub_name'].append('train')
            [results[f'{property_name} T-{time_steps - ii}'].append(hidden_state_valid.mean(0)[ii,0]) for ii in range(time_steps)]
            results['accuracy_train'].append(acc_trial_train)
            results['accuracy_test'].append(acc_trial_train)
        
            print('on test')
            for acc_trial_test in [0,1]:
                _idx_test, = np.where(acc_test == acc_trial_test)
                if len(_idx_test) > 1:
                    score_test = scoring_func(y_test[_idx_test],preds_test[_idx_test],
                                              confidence_range = confidence_range)
                    hidden_state_test,h_state_test,c_state_test = hidden_model.predict(X_test[_idx_test],
                                                                                   batch_size = batch_size,
                                                                                   verbose = 1)
                    print('{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_'.format(*list(hidden_state_test.mean(0).reshape(7,))))
                    
                    results['fold'].append(fold)
                    results['score'].append(np.mean(score_test))
                    results['n_sample'].append(X_test[_idx_test].shape[0])
                    results['source'].append('same')
                    results['sub_name'].append(np.unique(groups[test])[0])
                    results['accuracy_train'].append(acc_trial_train)
                    results['accuracy_test'].append(acc_trial_test)
                    [results[f'{property_name} T-{time_steps - ii}'].append(hidden_state_test.mean(0)[ii,0]) for ii in range(time_steps)]
                    
        gc.collect()
        
        results_to_save = pd.DataFrame(results)
        
        results_to_save.to_csv(csv_name,index = False)
