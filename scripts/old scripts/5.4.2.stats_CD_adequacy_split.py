#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:49:24 2021

@author: ning
"""

import os
import gc
import re
import utils

from glob import glob

import numpy   as np
import pandas  as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from sklearn import linear_model
from sklearn.model_selection import (LeaveOneGroupOut,
                                     permutation_test_score,
                                     cross_validate)

sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'adequacy' # confidence or adequacy
cv_type     = 'cross_domain' # LOO or cross_domain
decoder     = 'SVM' #
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../figures/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

working_data = glob(os.path.join(working_dir,'*csv'))
working_data = [item for item in working_data if ('past' in item) or ('recent' in item)]

# common settings
ylim            = (0.45,.8)
star_h          = 0.78
confidence_range= 4
time_steps      = 3
dict_rename     = {0:'incorrect trials',1:'correct trials'}
dict_condition  = {'past':'T-7,T-6,T-5','recent':'T-3,T-2,T-1'}
dict_source     = {'cognitive':'Cognitive',
                   'mem_4':'Memory',
                   'mixed_4':'Mixed'}
domains         = ['Cognitive','Memory','Mixed']
xargs           = dict(row          = 'source',
                       row_order    = list(dict_source.values()),
                       hue          = 'condition',
                       hue_order    = list(dict_condition.values()),
                       order        = ['SVM','RF','RNN'],
                       # col          = 'accuracy_test',
                       # col_order    = ['correct trials','incorrect trials',],
                       split        = True,
                       inner        = 'quartile',
                       cut          = 0,
                       scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       )

df = []
for f in working_data:
    temp            = pd.read_csv(f)
    temp['decoder'] = f.split('/')[-1].split('_')[0]
    condition = f.split('_')[-1].split(' ')[0]
    temp['condition'] = dict_condition[condition]
    df.append(temp)
df = pd.concat(df)

# average over folds
df_ave = df.groupby(['decoder','condition','source','filename']).mean().reset_index()
df_ave['source'] = df_ave['source'].map(dict_source)

# significance of scores
np.random.seed(12345)
results = dict(#accuracy_train   = [],
               #accuracy_test    = [],
               condition        = [],
               score_mean       = [],
               score_std        = [],
               ps               = [],
               decoder          = [],
               target_data      = [],
               )
for (_decoder,target_data,condition),df_sub in df_ave.groupby(
        ['decoder','source','condition']):
    scores = df_sub['score'].values
    
    gc.collect()
    ps = utils.resample_ttest(scores,
                              baseline      = .5,
                              n_ps          = 2,
                              n_permutation = int(1e4),
                              n_jobs        = -1,
                              verbose       = 1,
                              )
    gc.collect()
    
    # results['accuracy_train'].append(acc_train)
    # results['accuracy_test' ].append(acc_test)
    results['condition'     ].append(condition)
    results['score_mean'    ].append(np.mean(scores))
    results['score_std'     ].append(np.std(scores))
    results['ps'            ].append(np.mean(ps))
    results['decoder'       ].append(_decoder)
    results['target_data'   ].append(target_data)
results = pd.DataFrame(results)

temp = []
for target_data,df_sub in results.groupby(['target_data','condition']):
    df_sub                  = df_sub.sort_values(['ps'])
    pvals                   = df_sub['ps'].values
    converter               = utils.MCPConverter(pvals = pvals)
    d                       = converter.adjust_many()
    df_sub['ps_corrected']  = d['bonferroni'].values
    temp.append(df_sub)
results = pd.concat(temp)
results['stars']        = results['ps_corrected'].apply(utils.stars)
results.to_csv(os.path.join(stats_dir,'scores_split.csv'),index = False)

# plot scores
g = sns.catplot(x           = 'decoder',
                y           = 'score',
                data        = df_ave,
                kind        = 'violin',
                aspect      = 1.5,
                **xargs)
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
(g.set_axis_labels("","ROC AUC")
   .set_titles("{row_name}")
  .set(ylim = ylim)
  )
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
g._legend.set_title("")
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
## add stars
for ((target_data),results_sub),ax in zip(results.groupby(['target_data']),g.axes.flatten()):
    for ii,text_obj in enumerate(xtick_order):
        # print(target_data,_decoder)
        position        = text_obj.get_position()
        xtick_label     = text_obj.get_text()
        results_sub_stats = results_sub[results_sub['decoder'] == xtick_label].sort_values(['condition'],ascending = False)
        for (jj,temp_row),adjustment in zip(results_sub_stats.iterrows(),[-0.125,0.125]):
            # print(temp_row['stars'])
            if '*' in temp_row['stars']:
                ax.annotate(temp_row['stars'],
                            xy          = (position[0] + adjustment,star_h),
                            ha          = 'center',
                            fontsize    = 14)

g.savefig(os.path.join(figures_dir,'scores_split.jpg'),
          dpi = 300,
          bbox_inches = 'tight')

##############################################################################
# get the weights of the regression model
df_reg = df_ave[df_ave['decoder'] == decoder]

results = dict(#accuracy_train = [],
               condition = [],
               coefficient = [],
               ps = [],
               )
results[experiment] = []
slopes = dict(#accuracy_train = [],
              condition = [],
              coefficient = [],
              intercept = [],
              )
slopes[experiment] = []
# fit a regression to show the linear trend of the weights

for (condition),df_sub in df_reg.groupby(['condition']):
    for idx_confidence in range(confidence_range):
        weight_for_fit = df_sub[[f'weight T-{time_steps-ii} C-{idx_confidence}' for ii in range(time_steps)]].values
        xx = np.vstack([np.arange(time_steps) for _ in range(weight_for_fit.shape[0])])
        groups = df_sub['filename'].values
        groups = np.vstack([groups for _ in range(time_steps)]).T
        cv = LeaveOneGroupOut()
        pipeline = linear_model.BayesianRidge(fit_intercept = True)
        # permutation test to get p values
        _score,_,pval_slope = permutation_test_score(pipeline,
                                                     xx.reshape(-1,1),
                                                     weight_for_fit.reshape(-1,1).ravel(),
                                                     cv = cv,
                                                     groups = groups.reshape(-1,1).ravel(),
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
                              weight_for_fit.reshape(-1,1).ravel(),
                              groups = groups.reshape(-1,1).ravel(),
                              cv = cv,
                              n_jobs = -1,
                              verbose = 1,
                              scoring = 'neg_mean_squared_error',
                              return_estimator = True,
                              )
        gc.collect()
        coefficients = np.array([est.coef_[0] for est in _res['estimator']])
        intercepts = np.array([est.intercept_ for est in _res['estimator']])
        # results['accuracy_train'].append(acc_train)
        results['condition'].append(condition)
        results[experiment].append(idx_confidence)
        results['ps'].append(pval_slope)
        results['coefficient'].append(np.mean(coefficients))
        for coef_,inte_ in zip(coefficients,intercepts):
            # slopes['accuracy_train'].append(acc_train)
            slopes[experiment].append(idx_confidence)
            slopes['condition'].append(condition)
            slopes['coefficient'].append(coef_)
            slopes['intercept'].append(inte_)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'features_split.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'slopes_split.csv'),index = False)

df_weights = []
for (condition),df_sub in df_reg.groupby(['condition']):
    weights = df_sub[[col for col in df_sub.columns if ('weight' in col)]].values
    w = np.concatenate([[w.reshape(time_steps,-1).T] for w in weights])
    for ii in range(confidence_range):
        weight_for_plot = w[:,ii,:]
        temp = pd.DataFrame(weight_for_plot,columns = [f'T-{time_steps-ii}' for ii in range(time_steps)])
        temp[experiment] = ii + 1
        temp['condition'] = condition
        df_weights.append(temp)
df_weights = pd.concat(df_weights)
df_weights_plot = pd.melt(df_weights,
                          id_vars = ['condition',experiment],
                          value_vars = [f'T-{time_steps-ii}' for ii in range(time_steps)],
                          var_name = ['Time'],
                          value_name = 'Weights')

colors = ['blue','orange','green','red']
df_weights_plot['x'] = df_weights_plot['Time'].apply(lambda x:-int(x[-1]))

g = sns.lmplot(x = 'x',
               y = 'Weights',
               hue = experiment,
               col = 'condition',
               col_order = xargs['hue_order'],
               data = df_weights_plot,
               palette = colors,
               markers = '.',
               robust = False, # this takes all the time to run
               seed = 12345,
               aspect = 1.5,
               sharex = False,
               sharey = False,
               )
(g.set_titles("")
  .set_axis_labels('Trials back','Weights (A.U.)')
  .set(xlim = (-time_steps-0.5,-0.5),)
  )
for ((condition), df_sub),ax in zip(df_reg.groupby(['condition']),g.axes.T.flatten()):
    ax.set(**utils.set_line_lims(dict_condition,
                                 [(-0.1,0.1),
                                  (-0.2,0.2)])[condition])
g.savefig(os.path.join(figures_dir,'features_split.jpg'),
          dpi = 300,
          bbox_inches = 'tight')














