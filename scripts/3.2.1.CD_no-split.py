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
                   get_domains_maps,
                   pipeline_arguments,
                   pipelines,
                   model_fit,
                   model_prediction,
                   model_evaluation,
                   get_model_attributions,
                   get_feature_targets
                   )

from sklearn.model_selection import LeaveOneGroupOut

domains = np.array(list(get_domains_maps().values()))

target_attributes = 'confidence-accuracy' # change attributes
if True:
    for _idx_target,_idx_source in LeaveOneGroupOut().split(np.random.rand(4,10),np.random.rand(4),
                                                             groups = domains):
        
        model_name          = 'RF' # change model name
        reg_clf             = 'regression' # change type
        experiment_type     = 'cross_domain'
        split_data          = 'no-split'
        data_dir            = '../data'
        source_data         = domains[_idx_source][0]
        # print(source_data,target_attributes)
        is_rnn              = True if model_name == 'RNN' else False
        working_df_name     = os.path.join(data_dir,target_attributes,f'{source_data}.csv')
        saving_dir          = f'../results/{reg_clf}/{target_attributes}/{experiment_type}'
        batch_size          = 32
        n_features          = 7 if target_attributes != 'confidence-accuracy' else 14
        time_steps          = 7
        confidence_range    = 4
        n_jobs              = -1
        verbose             = 1
        debug               = True
        
        df_source           = pd.read_csv(working_df_name,)
        df_source           = check_column_type(df_source)
        features_source,targets_source,groups_source,accuracies_source = get_feature_targets(df_source,
                                                                                             n_features         = n_features,
                                                                                             time_steps         = time_steps,
                                                                                             target_attributes  = target_attributes,
                                                                                             group_col          = 'sub',
                                                                                             normalize_features = False,
                                                                                             normalize_targets  = False,)
        cv                  = LeaveOneGroupOut()
        
        idxs_train,idxs_test = [],[]
        for train,test in cv.split(features_source,targets_source,groups = groups_source):
            idxs_train.append(train)
            idxs_test.append(test)
        np.random.seed(12345)
        if len(idxs_train) > 50:
            _idx = np.random.choice(len(idxs_train),size = 50,replace = False)
            idxs_train = [idxs_train[ii] for ii in _idx]
        
        # train the decoder on all the source data
        # make the model
        xargs = pipeline_arguments()
        pipeline = pipelines(xargs)[f'{model_name.lower()}_{reg_clf}']
        # fit the model
        pipeline = model_fit(pipeline,
                             cv         = zip(idxs_train,idxs_test),
                             X_train    = features_source,
                             y_train    = targets_source,
                             model_name = model_name.lower(),
                             reg_clf    = reg_clf,
                             n_features = n_features,
                             )
        
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
            
            features_target,targets_target,groups_target,accuracies_target = get_feature_targets(df_target,
                                                                                                 n_features         = n_features,
                                                                                                 time_steps         = time_steps,
                                                                                                 target_attributes  = target_attributes,
                                                                                                 group_col          = 'sub',
                                                                                                 normalize_features = False,
                                                                                                 normalize_targets  = False,
                                                                                                 )
            
            csv_name    = os.path.join(saving_dir,f'results_{model_name}_{target_attributes}_{source_data}_{target_data}.csv')
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
            print(cv.get_n_splits(features_target,targets_target,groups=groups_target))
            
            # test the decoder on the target data for each subject
            for fold,(train_,test) in enumerate(cv.split(features_target,targets_target,groups=groups_target)):
                print(f'fold {fold}')
                if fold not in results['fold']:
                    # leave out test data
                    
                    X_test, y_test  = features_target[test],targets_target[test]
                    acc_test        = accuracies_target[test]
                    groups_test     = groups_target[test]
                    
                    # test the model
                    y_pred          = model_prediction(pipeline,X_test,reg_clf = reg_clf,is_rnn = is_rnn,)
                    # evaluate the model
                    CR_dim      = y_pred.shape[1] if reg_clf == 'classification' else 4
                    scores          = model_evaluation(y_test,y_pred,
                                                       confidence_range = CR_dim,
                                                       reg_clf          = reg_clf,
                                                       is_rnn           = is_rnn,
                                                       )
                    # get the weights
                    properties      = get_model_attributions(pipeline,X_test,y_test,
                                                             model_name = model_name.lower(),
                                                             reg_clf    = reg_clf,
                                                             )
                    # get parameters
                    params          = pipeline.best_estimator_.get_params()
                    
                    # save the results
                    if model_name == 'SVM' and reg_clf == 'classification':
                        for ii_row,row in enumerate(properties):
                            results['fold'                          ].append(fold)
                            results['score'                         ].append(scores)
                            results['n_sample'                      ].append(X_test.shape[0])
                            results['source'                        ].append('same')
                            results['sub_name'                      ].append(np.unique(groups_test)[0])
                            [results[f'features T-{n_features - ii_item}'].append(item) for ii_item,item in enumerate(row)]
                            results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
                            results['feature_type'                  ].append(target_attributes)
                            results['source_data'                   ].append(source_data)
                            results['target_data'                   ].append(target_data)
                            results['special'                       ].append(ii_row + 1)
                    else:
                        results['fold'                          ].append(fold)
                        results['score'                         ].append(scores)
                        results['n_sample'                      ].append(X_test.shape[0])
                        results['source'                        ].append('same')
                        results['sub_name'                      ].append(np.unique(groups_test)[0])
                        [results[f'features T-{n_features - ii}'].append(item) for ii,item in enumerate(properties)]
                        results['best_params'                   ].append('|'.join(f'{key}:{value}' for key,value in params.items()))
                        results['feature_type'                  ].append(target_attributes)
                        results['source_data'                   ].append(source_data)
                        results['target_data'                   ].append(target_data)
                        results['special'                       ].append('None')
                results_to_save = pd.DataFrame(results)
            else:
                results_to_save = pd.DataFrame(results)
            
            results_to_save.to_csv(csv_name,index = False)
