#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:49:24 2021

@author: ning
"""

import os
import gc
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

experiment  = 'confidence' # confidence or adequacy
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
ylim            = (0.36,.88)
star_h          = 0.78
confidence_range= 4
time_steps      = 3
dict_rename     = {0:'incorrect trials',1:'correct trials'}
dict_condition  = {'past':'T-7,T-6,T-5','recent':'T-3,T-2,T-1'}
dict_source     = {'cognitive':'Cognitive',
                   'mem_4':'Memory',
                   'mixed_4':'Mixed'}
domains         = ['Cognitive','Memory','Mixed']
xargs           = dict(row          = 'accuracy_train',
                       row_order    = ['correct trials','incorrect trials',],
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
col_order = [f'{cor}_{src}'for src in domains for cor in xargs['row_order'] ]
xargs['col_order'] = col_order

df = []
for f in working_data:
    temp            = pd.read_csv(f)
    temp['decoder'] = f.replace('\\','/').split('/')[-1].split('_')[0]
    condition = f.split('_')[-1].split(' ')[0]
    temp['condition'] = dict_condition[condition]
    df.append(temp)
df = pd.concat(df)

# average over folds
df_ave = df.groupby(['decoder','condition','source','filename','fold','accuracy_train','accuracy_test']).mean().reset_index()

for col_name in ['accuracy_train','accuracy_test']:
    df_ave[col_name] = df_ave[col_name].map(dict_rename)
df_ave['source'] = df_ave['source'].map(dict_source)

# significance of scores
np.random.seed(12345)
results = dict(accuracy_train   = [],
               accuracy_test    = [],
               score_mean       = [],
               score_std        = [],
               ps               = [],
               decoder          = [],
               target_data      = [],
               condition        = [],
               )
for (_decoder,acc_train,acc_test,target_data,condition),df_sub in df_ave.groupby(
        ['decoder','accuracy_train','accuracy_test','source','condition']):
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
    
    results['accuracy_train'].append(acc_train)
    results['accuracy_test' ].append(acc_test)
    results['score_mean'    ].append(np.mean(scores))
    results['score_std'     ].append(np.std(scores))
    results['ps'            ].append(np.mean(ps))
    results['decoder'       ].append(_decoder)
    results['target_data'   ].append(target_data)
    results['condition'     ].append(condition)
results = pd.DataFrame(results)

temp = []
for (target_data,condition),df_sub in results.groupby(['target_data','condition']):
    df_sub                  = df_sub.sort_values(['ps'])
    pvals                   = df_sub['ps'].values
    converter               = utils.MCPConverter(pvals = pvals)
    d                       = converter.adjust_many()
    df_sub['ps_corrected']  = d['bonferroni'].values
    temp.append(df_sub)
results = pd.concat(temp)
results['stars']        = results['ps_corrected'].apply(utils.stars)
results.to_csv(os.path.join(stats_dir,'scores_split.csv'),index = False)

paired = pd.read_csv(os.path.join(stats_dir,'scores_paired.csv'))
paired['target_data'] = paired['source'].values

specs = {name:xargs[name] for name in [
        'hue',
        'hue_order',
        'split',
        'inner',
        'cut',
        'scale',
        'order',
        'palette',]}
xtick_order = [plt.text(0, 0, 'SVM'),
               plt.text(1, 0, 'RF'),
               plt.text(2, 0, 'RNN')]
# plot scores
fig,axes = plt.subplots(figsize = (18,36),
                        nrows = 6,
                        ncols = 2,
                        sharex = True,
                        sharey = True,
                        )
for idx_ax,(((target_data,acc_train,acc_test,),df_sub),ax) in enumerate(zip(df_ave.groupby([
        'source',
        'accuracy_train',
        'accuracy_test',
        ]),axes.flatten())):
    ax = sns.violinplot(x = 'decoder',
                        y = 'score',
                        data = df_sub,
                        ax = ax,
                        **specs,)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().set_title('')
    if idx_ax != 0:
        ax.get_legend().remove()
    idx_stats = np.logical_and.reduce([
            results['accuracy_train'] == acc_train,
            results['accuracy_test' ] == acc_test,
            results['target_data'   ] == target_data,])
    results_sub = results[idx_stats]
    idx_pairs = np.logical_and.reduce([
            paired['accuracy_train'] == acc_train,
            paired['accuracy_test' ] == acc_test,
            paired['target_data'   ] == target_data,])
    pair_sub = paired[idx_pairs]
    
    for ii,text_obj in enumerate(xtick_order):
        # print(target_data,_decoder)
        position        = text_obj.get_position()
        xtick_label     = text_obj.get_text()
        results_sub_stats = results_sub[results_sub['decoder'] == xtick_label].sort_values(['condition'],ascending = False)
        for (jj,temp_row),adjustment in zip(results_sub_stats.iterrows(),[-0.125,0.125]):
            # print(temp_row['stars'])
            ax.annotate(temp_row['stars'],
                        xy          = (position[0] + adjustment,star_h),
                        ha          = 'center',
                        fontsize    = 14)
        pair_sub_sub = pair_sub[pair_sub['decoder'] == xtick_label]
        ax.plot([ii-adjustment,ii-adjustment,
                 ii+adjustment,ii+adjustment],
                [0.80,0.82,0.82,0.80],
                color = 'black')
        ax.annotate(pair_sub_sub['stars'].values[0].replace('*','o'),
                    xy = (ii,0.84),
                    ha = 'center',
                    va = 'center',
                    fontsize = 14,
                    )
    ax.set(ylim = ylim,
           xlabel = "",
           ylabel = "ROC AUC",
           title = f'{acc_train} -> {acc_test} | {target_data}',
           )
    ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black')
plt.subplots_adjust(top = 1.)

fig.savefig(os.path.join(figures_dir,'scores_split.jpg'),
            dpi = 300,
            bbox_inches = 'tight')

##############################################################################
# get the weights of the regression model
df_reg = df_ave[df_ave['decoder'] == decoder]

results = dict(accuracy_train = [],
               condition = [],
               coefficient = [],
               ps = [],
               )
results[experiment] = []
slopes = dict(accuracy_train = [],
              condition = [],
              coefficient = [],
              intercept = [],
              )
slopes[experiment] = []
# fit a regression to show the linear trend of the weights
for (acc_train,condition),df_sub in df_reg.groupby(['accuracy_train','condition']):
    for idx_confidence in range(confidence_range):
        weight_for_fit = df_sub[[f'weight T-{time_steps-ii} C-{idx_confidence}' for ii in range(time_steps)]].values
        xx = np.vstack([np.arange(time_steps) for _ in range(weight_for_fit.shape[0])])
        groups = df_sub['fold'].values
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
        results['accuracy_train'].append(acc_train)
        results['condition'].append(condition)
        results[experiment].append(idx_confidence)
        results['ps'].append(pval_slope)
        results['coefficient'].append(np.mean(coefficients))
        for coef_,inte_ in zip(coefficients,intercepts):
            slopes['accuracy_train'].append(acc_train)
            slopes['condition'].append(condition)
            slopes[experiment].append(idx_confidence)
            slopes['coefficient'].append(coef_)
            slopes['intercept'].append(inte_)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'features_split.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'slopes_split.csv'),index = False)

df_weights = []
for (acc_train,condition),df_sub in df_reg.groupby(['accuracy_train','condition']):
    weights = df_sub[[col for col in df_sub.columns if ('weight' in col)]].values
    w = np.concatenate([[w.reshape(time_steps,-1).T] for w in weights])
    for ii in range(confidence_range):
        weight_for_plot = w[:,ii,:]
        temp = pd.DataFrame(weight_for_plot,columns = [f'T-{time_steps-ii}' for ii in range(time_steps)])
        temp['accuracy_train'] = acc_train
        temp['condition'] = condition
        temp[experiment] = ii + 1
        df_weights.append(temp)
df_weights = pd.concat(df_weights)
df_weights_plot = pd.melt(df_weights,
                          id_vars = ['accuracy_train','condition',experiment],
                          value_vars = [f'T-{time_steps-ii}' for ii in range(time_steps)],
                          var_name = ['Time'],
                          value_name = 'Weights')

colors = ['blue','orange','green','red']
df_weights_plot['x'] = df_weights_plot['Time'].apply(lambda x:-int(x[-1]))

g = sns.lmplot(x = 'x',
                y = 'Weights',
                hue = experiment,
                col = 'accuracy_train',
                col_order = xargs['row_order'],
                row = 'condition',
                row_order = xargs['hue_order'],
                data = df_weights_plot,
                palette = colors,
                markers = '.',
                robust = False, # this takes all the time to run
                sharex = False,
                sharey = False,
                seed = 12345,
                aspect = 1.5,
                )
(g.set_titles("train on {col_name}")
  .set_axis_labels('Trials back','Weights (A.U.)')
  .set(xlim = (-time_steps - 0.5,-0.5),)
  )
for ((acc_train,condition), df_sub),ax in zip(df_reg.groupby(['accuracy_train','condition']),g.axes.T.flatten()):
    ax.set(**utils.set_line_lims(dict_condition,
                                 [(-0.25,0.25),
                                  (-0.65,0.65)])[condition])

g.savefig(os.path.join(figures_dir,'features_split.jpg'),
          dpi = 300,
          bbox_inches = 'tight')














