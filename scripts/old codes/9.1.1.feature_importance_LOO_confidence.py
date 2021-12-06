#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:30:26 2021

@author: ning
"""

import os
import re
import gc
import utils

from glob import glob

import numpy   as np
import pandas  as pd
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import (LeaveOneGroupOut,
                                     permutation_test_score,
                                     cross_validate)

sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'confidence' # confidence or adequacy
cv_type     = 'LOO' # LOO or cross_domain
decoder     = 'RF' #
confidence_range = 4
time_steps  = 7
xargs           = dict(hue          = 'accuracy_test',
                       hue_order    = ['correct trials','incorrect trials',],
                       col          = 'accuracy_train',
                       col_order    = ['correct trials','incorrect trials',],
                       # split        = True,
                       # inner        = 'quartile',
                       # cut          = 0,
                       # scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       # col_order    = ['SVM','RF','RNN'],
                       )
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../figures/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

working_data = glob(os.path.join(working_dir,
                                 '*.csv'))
working_data = [item for item in working_data if ('past' not in item) and ('recent' not in item)]
df_ave = utils.load_results(
    data_type      = 'confidence', # confidence or adequacy
    within_cross   = cv_type, # LOO or cross_domain
    working_data   = working_data,
    dict_rename    = {0:'incorrect trials',1:'correct trials'},
    dict_condition = None,
    )

# get the weights of the regression model
df_rf = df_ave[df_ave['decoder'] == decoder]
df_fi = df_rf[[f'feature importance T-{7-ii}' for ii in range(time_steps)]]
for col in ['study_name', 'accuracy_train', 'accuracy_test']:
    df_fi[col] = df_rf[col]

df_plot = pd.melt(df_fi,
                  id_vars = ['study_name', 'accuracy_train', 'accuracy_test'],
                  value_vars = [f'feature importance T-{7-ii}' for ii in range(time_steps)],
                  var_name = 'Time',
                  value_name = 'feature importance',
                  )
df_plot['x'] = df_plot['Time'].apply(lambda x: x.split(' ')[-1])

results = dict(accuracy_train = [],
               accuracy_test = [],
               coefficients = [],
               pval = [],
               )
slopes = dict(accuracy_train = [],
              accuracy_test = [],
              slopes = [],
              intercepts = [],
              )
# linear trend testing
for (acc_train,acc_test),df_sub in df_rf.groupby(['accuracy_train','accuracy_test']):
    feature_importance = df_sub[[f'feature importance T-{7-ii}' for ii in range(time_steps)]].values
    xx = np.vstack([np.arange(time_steps) for _ in range(feature_importance.shape[0])])
    groups = np.vstack([[ii] * xx.shape[1]] for ii in range(xx.shape[0]))
    cv = LeaveOneGroupOut()
    pipeline = linear_model.BayesianRidge(fit_intercept = True)
    # permutation test to get p values
    _score,_,pval_slope = permutation_test_score(pipeline,
                                                 xx.reshape(-1,1),
                                                 feature_importance.reshape(-1,1).ravel(),
                                                 groups = groups.reshape(-1,1).ravel(),
                                                 cv = cv,
                                                 n_jobs = -1,
                                                 random_state = 12345,
                                                 n_permutations = int(1e4),
                                                 scoring = 'neg_mean_squared_error',
                                                 verbose = 1,
                                                 )
    # cross validation to get the slopes and intercepts
    gc.collect()
    _res = cross_validate(pipeline,
                          xx.reshape(-1,1),
                          feature_importance.reshape(-1,1).ravel(),
                          groups = groups.reshape(-1,1).ravel(),
                          cv = cv,
                          n_jobs = -1,
                          verbose = 1,
                          scoring = 'neg_mean_squared_error',
                          return_estimator = True,
                          )
    gc.collect()
    results['accuracy_train'].append(acc_train)
    results['accuracy_test'].append(acc_test)
    results['pval'].append(pval_slope)
    coefficients = np.array([est.coef_[0] for est in _res['estimator']])
    intercepts = np.array([est.intercept_ for est in _res['estimator']])
    results['coefficients'].append(np.mean(coefficients))
    for coef_,inte_ in zip(coefficients,intercepts):
        slopes['accuracy_train'].append(acc_train)
        slopes['accuracy_test'].append(acc_test)
        slopes['slopes'].append(coef_)
        slopes['intercepts'].append(inte_)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'feature_importance.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'feature_importance_slopes.csv'),index = False)

groupby = ['accuracy_train', 'accuracy_test','Time']
permutations = {name:[] for name in groupby}
permutations['ps_mean'] = []
permutations['diff_mean'] = []
for _factors,df_sub in df_plot.groupby(groupby):
    x = df_sub['feature importance'].values
    ps = utils.resample_ttest(x,
                              baseline = 0,
                              n_ps = 10,
                              n_permutation = int(1e5),
                              one_tail = True,
                              n_jobs = -1,
                              verbose = 1,)
    for name,_name in zip(groupby,_factors):
        permutations[name].append(_name)
    permutations['ps_mean'].append(np.mean(ps))
    permutations['diff_mean'].append(np.mean(x))
permutations = pd.DataFrame(permutations)
permutations = permutations.sort_values('ps_mean')
ps = permutations['ps_mean'].values
converter = utils.MCPConverter(pvals = ps)
d = converter.adjust_many()
permutations['ps_corrected'] = d['bonferroni'].values
permutations['stars'] = permutations['ps_corrected'].apply(utils.stars)
permutations['Time'] = permutations['Time'].apply(lambda x:x.split(' ')[-1])

g = sns.catplot(x = 'x',
                y = 'feature importance',
                data = df_plot,
                kind = 'bar',
                aspect = 1.5,
                **xargs
                )

xx = np.linspace(0,time_steps-1,1000)
df_fi = df_rf[[f'feature importance T-{7-ii}' for ii in range(time_steps)]]
for acc_train, ax in zip(xargs['hue_order'],g.axes.flatten()):
    for acc_test,color in zip(xargs['hue_order'],xargs['palette']):
        df_sub = df_plot[np.logical_and(
                            df_plot['accuracy_train'] == acc_train,
                            df_plot['accuracy_test'] == acc_test)]
        df_sub['xx'] = df_sub['x'].apply(lambda x: time_steps - int(x[-1]))
        ax = sns.regplot(x = 'xx',
                         y = 'feature importance',
                         data = df_sub,
                         color = color,
                         marker = None,
                         scatter = False,
                         ax = ax,)
# add significance level
for ax,acc_train in zip(g.axes.flatten(),xargs['col_order']):
    df_sub = permutations[permutations['accuracy_train'] == acc_train]
    df_sub = df_sub.sort_values(['accuracy_train','accuracy_test'])
    xtick_order = list(ax.xaxis.get_majorticklabels())
    
    for ii,text_obj in enumerate(xtick_order):
        position        = text_obj.get_position()
        xtick_label     = text_obj.get_text()
        df_sub_stats    = df_sub[df_sub['Time'] == xtick_label].sort_values(['accuracy_test'])
        for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.1275,0.125]):
            print(temp_row['stars'])
            ax.annotate(temp_row['stars'],
                        xy          = (ii + adjustment,.081),
                        ha          = 'center',va = 'center',
                        fontsize    = 10)
(g.set_titles('Trained on {col_name}')
 .set_axis_labels('Trial','Feature importance')
 .set(ylim = (-0.005,0.083)))
g._legend.set_title("Testing data")

g.savefig(os.path.join(figures_dir,
                        'feature_importance.jpg'),
          dpi = 300,
          bbox_inches = 'tight')






