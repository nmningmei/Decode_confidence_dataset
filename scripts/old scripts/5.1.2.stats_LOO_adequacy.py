#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:36:16 2021

@author: ning

statistics of the results

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

experiment  = 'adequacy' # confidence or adequacy
cv_type     = 'LOO' # LOO or cross_domain
decoder     = 'SVM' #
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../figures/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

df = []
for f in glob(os.path.join(working_dir,'*csv')):
    temp            = pd.read_csv(f)
    study_name      = re.findall('\(([^)]+)',f)[0]
    temp['study_name'] = study_name
    temp['decoder'] = f.split('/')[-1].split(' ')[0]
    df.append(temp)
df = pd.concat(df)

# common settings
ylim    = {'confidence':(0.45,0.9),
           'adequacy':(0.45,0.8)}[experiment]
star_h          = 0.75
confidence_range= 4
time_steps      = 7
dict_rename     = {0:'incorrect trials',1:'correct trials'}
xargs           = dict(hue          = 'decoder',
                       hue_order    = ['SVM','RF','RNN'],
                       # split        = True,
                       inner        = 'quartile',
                       cut          = 0,
                       # scale        = 'width',
                       palette      = ['gold','deepskyblue','tomato'],
                       )


# averge within each study
df_ave = df.groupby(['decoder',
                     'study_name',]).mean().reset_index()

# significance of scores
np.random.seed(12345)
results = dict(#accuracy_train   = [],
               #accuracy_test    = [],
               score_mean       = [],
               score_std        = [],
               ps               = [],
               decoder          = [],
               )
for (_decoder),df_sub in df_ave.groupby(['decoder',]):
    scores = df_sub['score'].values
    
    gc.collect()
    ps = utils.resample_ttest(scores,
                              baseline      = .5,
                              n_ps          = 100,
                              n_permutation = int(1e4),
                              n_jobs        = -1,
                              verbose       = 1,
                              )
    gc.collect()
    
    # results['accuracy_train'].append(acc_train)
    # results['accuracy_test' ].append(acc_test)
    results['score_mean'    ].append(np.mean(scores))
    results['score_std'     ].append(np.std(scores))
    results['ps'            ].append(np.mean(ps))
    results['decoder'       ].append(_decoder)
results = pd.DataFrame(results)

results                 = results.sort_values(['ps'])
pvals                   = results['ps'].values
converter               = utils.MCPConverter(pvals = pvals)
d                       = converter.adjust_many()
results['ps_corrected'] = d['bonferroni'].values
results['stars']        = results['ps_corrected'].apply(utils.stars)
results.to_csv(os.path.join(stats_dir,'scores.csv'),index = False)

# plot scores
df_ave['x'] = 0
g = sns.catplot(x           = 'x',
                y           = 'score',
                data        = df_ave,
                kind        = 'violin',
                aspect      = 1.5,
                **xargs)
(g.set_axis_labels("","ROC AUC")
  .set(ylim = ylim,
       xticklabels = [],))
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
# g._legend.remove()
g._legend.set_title("Models")

## add stars
ax = g.axes.flatten()[0]
for (jj,temp_row),adjustment in zip(results.iterrows(),[-0.25,0,0.25]):
    if '*' in temp_row['stars']:
        ax.annotate(temp_row['stars'],
                    xy = (adjustment, star_h),
                    ha = 'center',
                    fontsize = 14,
                    )
g.savefig(os.path.join(figures_dir,'scores.jpg'),
          dpi = 300,
          bbox_inches = 'tight')

# get the weights of the regression model
df_reg = df_ave[df_ave['decoder'] == decoder]

results = dict(#accuracy_train = [],
               coefficient = [],
               ps = [],
               )
results[experiment] = []
slopes = dict(#accuracy_train = [],
              coefficient = [],
              intercept = [],
              )
slopes[experiment] = []
# fit a regression to show the linear trend of the weights
for idx_confidence in range(confidence_range):
    weight_for_fit = df_reg[[f'weight T-{time_steps-ii} C-{idx_confidence}' for ii in range(time_steps)]].values
    xx = np.vstack([np.arange(time_steps) for _ in range(weight_for_fit.shape[0])])
    groups = df_reg['fold'].values
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
    results[experiment].append(idx_confidence)
    results['ps'].append(pval_slope)
    results['coefficient'].append(np.mean(coefficients))
    for coef_,inte_ in zip(coefficients,intercepts):
        # slopes['accuracy_train'].append(acc_train)
        slopes[experiment].append(idx_confidence)
        slopes['coefficient'].append(coef_)
        slopes['intercept'].append(inte_)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'features.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'slopes.csv'),index = False)

df_weights = []
    # df_sub = df_reg[df_reg['accuracy_train'] == acc_train]
weights = df_reg[[f'weight T-{time_steps-ii} C-{jj}' for ii in range(time_steps) for jj in range(confidence_range)]]
w = weights.values.reshape((-1,time_steps,confidence_range))
for ii in range(confidence_range):
    weight_for_plot = w[:,:,ii]
    temp = pd.DataFrame(weight_for_plot,columns = [f'T-{time_steps-ii}' for ii in range(time_steps)])
    # temp['accuracy_train'] = acc_train
    temp[experiment] = ii + 1
    df_weights.append(temp)
df_weights = pd.concat(df_weights)
df_weights_plot = pd.melt(df_weights,
                          id_vars = [experiment],
                          value_vars = [f'T-{time_steps-ii}' for ii in range(time_steps)],
                          var_name = ['Time'],
                          value_name = 'Weights')

colors = ['blue','orange','green','red']
df_weights_plot['x'] = df_weights_plot['Time'].apply(lambda x:-int(x[-1]))

g = sns.lmplot(x = 'x',
               y = 'Weights',
               hue = experiment,
               # col = 'accuracy_train',
               # col_order = xargs['hue_order'],
               data = df_weights_plot,
               palette = colors,
               markers = '.',
               robust = False, # this takes all the time to run
               seed = 12345,
               aspect = 1.5,
               )
(g.set_titles("")
  .set_axis_labels('Trial','Weights (A.U.)')
  .set(xlim = (-7.5,-0.5),
       xticks = np.arange(-7,0),
       xticklabels = [f'T-{time_steps-ii}' for ii in range(time_steps)])
  )
g.savefig(os.path.join(figures_dir,'features.jpg'),
          dpi = 300,
          bbox_inches = 'tight')






