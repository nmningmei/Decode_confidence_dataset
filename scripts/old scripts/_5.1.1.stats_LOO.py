#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 06:34:36 2021

@author: nmei
"""

import os
import gc
import utils

import numpy as np
import pandas as pd

from glob import glob
from sklearn.preprocessing import MinMaxScaler as scaler
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (LeaveOneGroupOut,
                                     permutation_test_score,
                                     cross_validate)
from itertools import combinations


experiment = 'adequacy' # confidence or adequacy
_decoder = 'regression'
working_dir = f'../results/{experiment}/LOO/'
stats_dir = f'../stats/{experiment}/LOO_compare_RNN_RF/'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
working_data = np.sort(glob(os.path.join(working_dir,'*.csv')))

df = []
for f in working_data:
    temp = pd.read_csv(f)
    decoder = f.split('/')[-1].split(' ')[0]
    experiment = f.split('/')[-1].split('(')[-1].split(')')[0]
    temp['model'] = decoder
    temp['experiment'] = experiment
    col_to_rename = [item for item in temp.columns if ('T-' in item)]
    rename_mapper = {item:f'{item.split(" ")[-1]}' for item in col_to_rename}
    temp = temp.rename(columns = rename_mapper)
    if True:#decoder == _decoder:
    # normalize with each decoding
#    temp_array = temp[[item for item in temp.columns if ('T-' in item)]].values
#    if decoder == 'RNN':
#        temp_array = np.abs(temp_array)
#    temp_array = scaler().fit_transform(temp_array.T)
#    temp[[item for item in temp.columns if ('T-' in item)]] = temp_array.T
        df.append(temp)
df = pd.concat(df)

df_plot = df[df['source'] != 'train']

# further process the data for plotting
df_plot['acc_train']  = df_plot['accuracy_train'].map({0:'incorrect',1:'correct'})
df_plot['acc_test'] = df_plot['accuracy_test'].map({0:'incorrect',1:'correct'})
df_plot['condition'] = df_plot['model'] + '_' + df_plot['acc_train'] + '_' + df_plot['acc_test']

res_scores = dict(experiment = [],
                  condition = [],
                  score_mean = [],
                  score_std = [],
                  pval = [],
                  )
res_features = dict(experiment = [],
                    condition = [],
                    slope = [],
                    intercept = [],
#                    slope_mean = [],
#                    slope_std = [],
#                    intercept_mean = [],
#                    intercept_std = [],
                    cv_score = [],
                    pval = [],
#                    y_mean = [],
#                    y_std = [],
                    )
res_slopes = dict(experiment = [],
                  condition = [],
                  slope = [],
                  intercept = [],
                  pval = [],)
for (experiment,condition),df_sub in df_plot.groupby(['experiment','condition']):
    # on the scores: compare against to theorectial chance level
    scores = df_sub['score'].values
    gc.collect()
    ps = utils.resample_ttest(scores,
                              0.5,
                              one_tail = True,
                              n_permutation = int(1e5),
                              n_jobs = -1,
                              verbose = 0,)
    gc.collect()
    res_scores['experiment'].append(experiment)
    res_scores['condition'].append(condition)
    res_scores['score_mean'].append(np.mean(scores))
    res_scores['score_std'].append(np.std(scores))
    res_scores['pval'].append(np.mean(ps))
    
    # on the feature contributions
    features = df_sub[[f'T-{7 - ii}' for ii in range(7)]].values
#    features = np.abs(features)
    xx = np.vstack([np.arange(7) for _ in range(features.shape[0])])
    groups = np.repeat(np.arange(features.shape[0]),7)
    cv = LeaveOneGroupOut()
    
    # a regularized linear regression
#    pipeline = linear_model.RidgeCV(alphas = np.logspace(-9,9,19),
#                                    scoring = 'neg_mean_squared_error',
#                                    cv = None,# set to None for efficient LOO algorithm
#                                    )
    pipeline = linear_model.BayesianRidge(fit_intercept = True)
    # permutation test to get p values
    _score,_,pval_slope = permutation_test_score(pipeline,xx.reshape(-1,1),features.reshape(-1,1).ravel(),
                                          cv = cv,
                                          groups = groups,
                                          n_jobs = -1,
                                          random_state = 12345,
                                          n_permutations = int(1e4),
                                          scoring = 'neg_mean_squared_error',
                                          verbose = 1,
                                          )
    # cross validation to get the slopes and intercepts
    gc.collect()
    _res = cross_validate(pipeline,xx.reshape(-1,1),features.reshape(-1,1).ravel(),
                          groups = groups,
                          cv = cv,
                          n_jobs = -1,
                          verbose = 1,
                          scoring = 'neg_mean_squared_error',
                          return_estimator = True,
                          )
    gc.collect()
    coefficients = np.array([est.coef_[0] for est in _res['estimator']])
    intercepts = np.array([est.intercept_ for est in _res['estimator']])
    # pval = utils.resample_ttest(coefficients,
    #                             baseline = 0,
    #                             n_ps = 10,
    #                             n_permutation = int(1e5),
    #                             one_tail = True,
    #                             n_jobs = -1,
    #                             verbose = 1,)
    # save
    for coef,interc in zip(coefficients,intercepts):
        res_slopes['experiment'].append(experiment)
        res_slopes['condition'].append(condition)
        res_slopes['slope'].append(coef)
        res_slopes['intercept'].append(interc)
        res_slopes['pval'].append(pval_slope)
    intercepts = np.array([est.intercept_ for est in _res['estimator']])
#    xxx = np.linspace(0,6,1000)
#    temp = np.array([est.predict(xxx.reshape(-1,1),) for est in _res['estimator']])
#    y_mean = temp[:,0,:]
#    y_std = temp[:,0,:]
    res_features['experiment'].append(experiment)
    res_features['condition'].append(condition)
    res_features['slope'].append([item for item in coefficients])
#    res_features['slope_std'].append(np.std(coefficients))
    res_features['intercept'].append([item for item in intercepts])
#    res_features['intercept_std'].append(np.std(intercepts))
    res_features['pval'].append(pval_slope)
    res_features['cv_score'].append(np.mean(_res['test_score']))
#    res_features['y_mean'].append(y_mean.mean(0))
#    res_features['y_std'].append(y_std.mean(0))
res_scores = pd.DataFrame(res_scores)
res_features = pd.DataFrame(res_features)
res_slopes = pd.DataFrame(res_slopes)

temp = []
for condition,df_sub in res_scores.groupby(['condition']):
    df_sub = df_sub.sort_values(['pval'])
    pvals = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res_scores = pd.concat(temp)

temp = []
for condition,df_sub in res_features.groupby(['condition']):
    df_sub = df_sub.sort_values(['pval'])
    pvals = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res_features = pd.concat(temp)

temp = np.concatenate(res_slopes['condition'].apply(lambda x:np.array(x.split('_'))).values).reshape(-1,3)
res_slopes['model'] = temp[:,0]
res_slopes['accuracy_train'] = temp[:,1]
res_slopes['accuracy_test'] = temp[:,-1]

temp = []
for condition, df_sub in res_slopes.groupby(['condition']):
    df_sub = df_sub.sort_values(['pval'])
    pvals = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res_slopes = pd.concat(temp)

# feature_comparison = dict(condition = [],
#                           experiment = [],
#                           pval = [],
#                           slope_rf_mean = [],
#                           slope_rf_std = [],
#                           slope_rnn_mean = [],
#                           slope_rnn_std = [],
#                           )
# for experiment,df_sub in res_slopes.groupby(['experiment']):
#     for conditions in list(combinations(pd.unique(res_slopes['condition']),2)):
#         a,b = conditions
#         if a[:3] != b[:3]:
#             for temp in conditions:
#                 if 'RF' in temp:
#                     rf = df_sub[df_sub['condition'] == temp]['slope'].values
#                 else:
#                     rnn = df_sub[df_sub['condition'] == temp]['slope'].values
#             ps = utils.resample_ttest_2sample(rf,rnn,
#                                               one_tail = False,
#                                               match_sample_size = False,
#                                               n_jobs = -1,
#                                               n_ps = 2,
#                                               n_permutation = int(1e5),
#                                               verbose = 1)
#             feature_comparison['experiment'].append(experiment)
#             feature_comparison['condition'].append(f'{a}_{b}')
#             feature_comparison['pval'].append(np.mean(ps))
#             feature_comparison['slope_rf_mean'].append(rf.mean())
#             feature_comparison['slope_rf_std'].append(rf.std())
#             feature_comparison['slope_rnn_mean'].append(rf.mean())
#             feature_comparison['slope_rnn_std'].append(rf.std())
#             del rf
#             del rnn
# feature_comparison = pd.DataFrame(feature_comparison)

res_scores['stars'] = res_scores['p_corrected'].apply(utils.stars)
res_scores.to_csv(os.path.join(stats_dir,'scores.csv'),index = False)

res_features['stars'] = res_features['p_corrected'].apply(utils.stars)
res_features.to_csv(os.path.join(stats_dir,'features.csv'),index = False)

res_slopes.to_csv(os.path.join(stats_dir,'slopes.csv'),index = False)

# feature_comparison.to_csv(os.path.join(stats_dir,'slope_comparison.csv'),index = False)

















