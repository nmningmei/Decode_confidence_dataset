#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:09:46 2021

@author: nmei

"""
import os,gc

import pandas as pd
import numpy as np

from utils import (check_column_type,
                   get_feature_targets,
                   build_RF,
                   get_domains_maps)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import explained_variance_score,r2_score
from sklearn.inspection import permutation_importance

domains = np.array(list(get_domains_maps().values()))

for target_attributes in ['confidence','accuracy','confidence-accuracy']:
    for _idx_target,_idx_source in LeaveOneGroupOut().split(np.random.rand(4,10),np.random.rand(4),
                                                             groups = domains):
        model_name          = 'RF'
        experiment_type     = 'cross_domain'
        # target_attributes   = 'confidence-accuracy' # change folder name
        split_data          = 'no-split'
        data_dir            = '../data'
        source_data         = domains[_idx_source][0]
        
        working_df_name     = os.path.join(data_dir,target_attributes,f'{source_data}.csv')
        saving_dir          = f'../results/{target_attributes}/{experiment_type}'
        batch_size          = 32
        n_features          = 7 if target_attributes != 'confidence-accuracy' else 14
        time_steps          = 7
        confidence_range    = 4
        n_jobs              = -1
        verbose             = 1
        debug               = True
        
        df_source           = pd.read_csv(working_df_name,)
        df_source           = check_column_type(df_source)
        df_source['temp']   = df_source['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
        df_source['sub']    = df_source['temp'] + '-' + df_source['sub'].astype(str)
        features_source,targets_source,groups_source,accuracies_source = get_feature_targets(df_source,
                                                                                             n_features         = n_features,
                                                                                             time_steps         = time_steps,
                                                                                             target_attributes  = target_attributes,
                                                                                             group_col          = 'sub',
                                                                                             normalize_features = False,
                                                                                             normalize_targets  = True,)
        cv                  = LeaveOneGroupOut()
        
        idxs_train,idxs_test = [],[]
        for train,test in cv.split(features_source,targets_source,groups = groups_source):
            idxs_train.append(train)
            idxs_test.append(test)
        if len(idxs_train) > 300:
            _idx = np.random.choice(len(idxs_train),size = 300,replace = False)
            idxs_train = [idxs_train[ii] for ii in _idx]
        
        # train the decoder on all the source data
        # make the model
        pipeline = make_pipeline(StandardScaler(),
                                 build_RF(bootstrap = True,
                                          oob_score = False,))
        
        model = GridSearchCV(pipeline,
                            {'randomforestregressor__n_estimators':np.logspace(0,3,4).astype(int),
                             'randomforestregressor__max_depth':np.arange(n_features) + 1},
                             scoring    = 'explained_variance',
                             n_jobs     = -1,
                             cv         = zip(idxs_train,idxs_test),
                             verbose    = 1,
                             )
        gc.collect()
        # train the model and validate the model
        model.fit(features_source,targets_source)
        gc.collect()
        
        # a = []
        for target_data in domains[_idx_target]:
            model_dir           = '../models/{}_{}_{}_{}_{}_{}'.format(*[
                                    model_name,
                                    experiment_type,
                                    target_attributes,
                                    split_data,
                                    source_data,
                                    target_data,
                                    ])
            
            df_target           = pd.read_csv(os.path.join(data_dir,target_attributes,f'{target_data}.csv'))
            df_target           = check_column_type(df_target)
            df_target['temp']   = df_target['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
            df_target['sub']    = df_target['temp'] + '-' + df_target['sub'].astype(str)
            
            features_target,targets_target,groups_target,accuracies_target = get_feature_targets(df_target,
                                                                                                 n_features         = n_features,
                                                                                                 time_steps         = time_steps,
                                                                                                 target_attributes  = target_attributes,
                                                                                                 group_col          = 'sub',
                                                                                                 normalize_features = False,
                                                                                                 normalize_targets  = True,
                                                                                                 )
            
            csv_name    = os.path.join(saving_dir,f'results_{model_name}_{target_attributes}_{source_data}_{target_data}.csv')
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
            print(cv.get_n_splits(features_target,targets_target,groups=groups_target))
            
            # test the decoder on the target data for each subject
            for fold,(train_,test) in enumerate(cv.split(features_target,targets_target,groups=groups_target)):
                print(f'fold {fold}')
                if fold not in results['fold']:
                    # leave out test data
                    
                    X_test, y_test  = features_target[test]  ,targets_target[test]
                    acc_test        = accuracies_target[test]
                    
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
                    gc.collect()
                    # get parameters
                    params = model.best_estimator_.get_params()
                    
                    # save the results
                    results['fold'                          ].append(fold)
                    results['score'                         ].append(scores)
                    results['r2'                            ].append(r2_score(y_test,y_pred,))
                    results['n_sample'                      ].append(X_test.shape[0])
                    results['source'                        ].append('different')
                    results['sub_name'                      ].append(X_test.shape[0])
                    [results[f'features T-{n_features - ii}'].append(item) for ii,item in enumerate(properties['importances_mean'])]
                    results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
                    results['feature_type'                  ].append(target_attributes)
                    results['source_data'                   ].append(source_data)
                    results['target_data'                   ].append(target_data)
                    
                results_to_save = pd.DataFrame(results)
                results_to_save.to_csv(csv_name,index = False)
