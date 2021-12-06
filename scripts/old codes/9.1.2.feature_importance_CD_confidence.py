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

experiment      = 'confidence' # confidence or adequacy
cv_type         = 'cross_domain' # LOO or cross_domain
decoder         = 'RF' #
confidence_range= 4
time_steps      = 7
domains         = {a:b for a,b in zip(['cognitive','mem_4','mixed_4'],
                                      ['Cognitive','Memory','Mixed'])
                       }
xargs           = dict(hue          = 'accuracy_test',
                       hue_order    = ['correct trials','incorrect trials',],
                       col          = 'accuracy_train',
                       col_order    = ['correct trials','incorrect trials',],
                       row_order    = list(domains.values()),
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
    data_type      = experiment, # confidence or adequacy
    within_cross   = cv_type, # LOO or cross_domain
    working_data   = working_data,
    dict_rename    = {0:'incorrect trials',1:'correct trials'},
    dict_condition = None,
    )
df_ave['source'] = df_ave['source'].map(domains)
# get the weights of the regression model
df_rf = df_ave[df_ave['decoder'] == decoder]
df_fi = df_rf[[f'feature importance T-{7-ii}' for ii in range(time_steps)]]
for col in ['fold', 'source', 'accuracy_train', 'accuracy_test']:
    df_fi[col] = df_rf[col]

df_plot = pd.melt(df_fi,
                  id_vars = ['fold', 'source', 'accuracy_train', 'accuracy_test'],
                  value_vars = [f'feature importance T-{7-ii}' for ii in range(time_steps)],
                  var_name = 'Time',
                  value_name = 'feature importance',
                  )
df_plot['x'] = df_plot['Time'].apply(lambda x: x.split(' ')[-1])
df_plot['xx'] = df_plot['x'].apply(lambda x: time_steps - int(x[-1]))

results = dict(accuracy_train = [],
               accuracy_test = [],
               coefficients = [],
               target_data = [],
               pval = [],
               )
slopes = dict(accuracy_train = [],
              accuracy_test = [],
              slopes = [],
              intercepts = [],
              target_data = []
              )
# linear trend testing
for (target_data,acc_train,acc_test),df_sub in df_plot.groupby(['source','accuracy_train','accuracy_test']):
    # feature_importance = df_sub[[f'feature importance T-{7-ii}' for ii in range(time_steps)]].values
    xx = df_sub['xx'].values
    feature_importance = df_sub['feature importance'].values
    groups = df_sub['fold'].values
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
    results['target_data'].append(target_data)
    coefficients = np.array([est.coef_[0] for est in _res['estimator']])
    intercepts = np.array([est.intercept_ for est in _res['estimator']])
    results['coefficients'].append(np.mean(coefficients))
    for coef_,inte_ in zip(coefficients,intercepts):
        slopes['accuracy_train'].append(acc_train)
        slopes['accuracy_test'].append(acc_test)
        slopes['slopes'].append(coef_)
        slopes['intercepts'].append(inte_)
        slopes['target_data'].append(target_data)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'feature_importance.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'feature_importance_slopes.csv'),index = False)

groupby = ['accuracy_train', 'accuracy_test','Time','source']
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
temp = []
for _,df_sub in permutations.groupby('source'):
    df_sub = df_sub.sort_values('ps_mean')
    ps = df_sub['ps_mean'].values
    converter = utils.MCPConverter(pvals = ps)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
permutations = pd.concat(temp)
permutations['stars'] = permutations['ps_corrected'].apply(utils.stars)
permutations['Time'] = permutations['Time'].apply(lambda x:x.split(' ')[-1])

g = sns.catplot(x = 'x',
                y = 'feature importance',
                row = 'source',
                data = df_plot,
                kind = 'bar',
                aspect = 1.5,
                **xargs
                )
for ax,((target_data,acc_train),df_sub) in zip(g.axes.flatten(),df_plot.groupby(['source','accuracy_train'])):
    print(ax.title.get_text(),target_data,acc_train)
    for acc_test,color in zip(xargs['hue_order'],xargs['palette']):
        df_sub_sub = df_sub[df_sub['accuracy_test'] == acc_test]
        ax = sns.regplot(x = 'xx',
                         y = 'feature importance',
                         scatter = False,
                         color = color,
                         data = df_sub_sub,
                         ax = ax,
                         )
# add significance level
for ax,((target_data,acc_train),df_sub) in zip(g.axes.flatten(),permutations.groupby(['source','accuracy_train'])):
    print(ax.title.get_text(),target_data,acc_train)
    # df_sub = permutations[permutations['accuracy_train'] == acc_train]
    df_sub = df_sub.sort_values(['accuracy_train','accuracy_test'])
    xtick_order = [plt.Text(0, 0, 'T-7'),
                   plt.Text(1, 0, 'T-6'),
                   plt.Text(2, 0, 'T-5'),
                   plt.Text(3, 0, 'T-4'),
                   plt.Text(4, 0, 'T-3'), 
                   plt.Text(5, 0, 'T-2'),
                   plt.Text(6, 0, 'T-1')]
    
    for ii,text_obj in enumerate(xtick_order):
        position        = text_obj.get_position()
        xtick_label     = text_obj.get_text()
        df_sub_stats    = df_sub[df_sub['Time'] == xtick_label].sort_values(['accuracy_test'])
        for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.1275,0.125]):
            print(temp_row['stars'])
            ax.annotate(temp_row['stars'],
                        xy          = (ii + adjustment,.044),
                        ha          = 'center',va = 'center',
                        fontsize    = 10)
(g.set_titles('Trained on {col_name} - {row_name}')
 .set_axis_labels('Trial','Feature importance')
 .set(ylim = (-0.005,0.047)))
g._legend.set_title("Testing data")

g.savefig(os.path.join(figures_dir,
                        'feature_importance.jpg'),
                      dpi = 300,
                      bbox_inches = 'tight')

















