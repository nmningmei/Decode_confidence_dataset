#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:39:57 2019

@author: nmei
"""

import os
import gc
from glob import glob
import pandas as pd
import numpy as np
from scipy import stats
from functools import partial
#from sklearn.preprocessing import MinMaxScaler as scaler
from utils import resample_ttest,stars
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')

experiment = 'adequacy' # confidence or adequacy
_decoder = 'regression'
working_dir = f'../results/{experiment}/LOO/'
stats_dir = f'../stats/{experiment}/LOO_compare_RNN_RF/'
figure_dir = f'../figures/{experiment}/LOO'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
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
    if decoder == _decoder:
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
df_plot['condition'] = df_plot['acc_train'] + '->' + df_plot['acc_test']
df_plot = df_plot.sort_values(['experiment','model','condition'])

df_stat_score = pd.read_csv(os.path.join(stats_dir,'scores.csv'))
df_stat_score['model'] = df_stat_score['condition'].apply(lambda x:x.split('_')[0])
df_stat_score = df_stat_score[df_stat_score['model'] == _decoder]
df_stat_score['acc_train'] = df_stat_score['condition'].apply(lambda x:x.split('_')[1])
df_stat_score['acc_test'] = df_stat_score['condition'].apply(lambda x:x.split('_')[-1])

df_stat_slope = pd.read_csv(os.path.join(stats_dir,'slopes.csv'))
df_stat_slope['model'] = df_stat_slope['condition'].apply(lambda x:x.split('_')[0])
df_stat_slope = df_stat_slope[df_stat_slope['model'] == _decoder]

df_stat_features = pd.read_csv(os.path.join(stats_dir,'features.csv'))
df_stat_features['model'] = df_stat_features['condition'].apply(lambda x:x.split('_')[0])
df_stat_features = df_stat_features[df_stat_features['model'] == _decoder]
df_stat_features['acc_train'] = df_stat_features['condition'].apply(lambda x: x.split('_')[-2])
df_stat_features['acc_test'] = df_stat_features['condition'].apply(lambda x: x.split('_')[-1])

xargs = dict(hue = 'acc_test',
#             hue_order = ['RF_correct','RF_incorrect','RNN_correct','RNN_incorrect'],
             hue_order = ['correct','incorrect',],
             split = True,
             inner = 'quartile',
             cut = 0,
             scale = 'width',
             palette = ['deepskyblue','tomato'])

fig,axes = plt.subplots(figsize = (16 * 2,26),
                      ncols = 2,)
for (acc_train,df_plot_sub),(_,df_stat_score_sub),ax in zip(df_plot.groupby(['acc_train']),
                                      df_stat_score.groupby('acc_train'),
                                      axes.flatten()):
    ax = sns.violinplot(y = 'experiment',
                        x = 'score',
                        data = df_plot_sub,
                        ax = ax,
                        **xargs)
    handles,labels = ax.get_legend_handles_labels()
    ytick_order = list(ax.yaxis.get_majorticklabels())
    
    for ii,text_obj in enumerate(ytick_order):
        position = text_obj.get_position()
        ytick_label = text_obj.get_text()
        df_sub_stats = df_stat_score_sub[df_stat_score_sub['experiment'] == ytick_label].sort_values(['condition'])
        for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.125,0.125]):
            if '*' in temp_row['stars']:
                print(temp_row['stars'])
                ax.annotate(temp_row['stars'],
                            rotation = 90,
                            xy = (1.01,ii + adjustment,),
                            verticalalignment = 'center',
                            fontsize = 14)
    ax.set(xlim = (0.15,1.05),xlabel = 'ROC AUC',ylabel = 'Study',title = f'Trained on {acc_train} trials')
    ax.axvline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
    ax.legend(handles = handles,labels = labels,loc = 'upper left',title = 'Test data')
ax.set(yticklabels = [],ylabel = '')
fig.savefig(os.path.join(figure_dir,
                         'scores.jpg'),
#            dpi = 400,
            bbox_inches = 'tight')

df_for_plot = df_plot.melt(id_vars = ['sub_name','acc_train','acc_test','experiment',],
                           value_vars = ['T-7', 'T-6', 'T-5', 'T-4', 'T-3', 'T-2', 'T-1'],)

fig,ax = plt.subplots(figsize = (16,16))
_xargs = dict(x = 'variable',
              y = 'value',
              hue = xargs['hue'],
              hue_order = xargs['hue_order'],
              data = df_for_plot,
              ax = ax,)
ax = sns.stripplot(dodge = True,
                   alpha = 1 / df_for_plot.shape[0] * 100,
                   **_xargs,)
# compute a less bias mean
temp_func = partial(stats.trim_mean,**dict(proportiontocut=0.05))
ax = sns.pointplot(dodge = 0.4,
                   palette = 'dark',
                   estimator = temp_func,
                   markers = ['D','d'],
                   join = False,
                   ci = None,
                   scale = 0.5,
                   **_xargs,
                   )
########################################################################
df_for_plot['x'] = df_for_plot['variable'].map({f'T-{7-ii}':ii for ii in range(7)})
df_for_plot = df_for_plot.sort_values(['acc'])
df_stat_features = df_stat_features.sort_values(['acc'])
df_stat_slope = df_stat_slope.sort_values(['acc'])

for (_condition,df_for_plot_sub),(_,df_stat_features_sub),(_,df_stat_slope_sub),_color in zip(
                        df_for_plot.groupby(['acc']),
                        df_stat_features.groupby(['acc']),
                        df_stat_slope.groupby(['acc']),
                        ['blue','orange','green','red']):
    x_vals = df_for_plot_sub['x'].values
    slopes = df_stat_slope_sub['slope'].values
    intercepts = df_stat_slope_sub['intercept'].values
    xxx = np.linspace(0,6,1000)
    yyy = xxx.reshape(-1,1).dot(slopes.reshape(1,-1)) + intercepts
    
    y_mean = temp_func(yyy,axis = 1)
    y_std = np.std(yyy,axis = 1) / np.sqrt(1000)
    ax.plot(xxx,y_mean,linestyle = '--',color = _color,alpha = 0.7)
    ax.fill_between(xxx,
                    y_mean + y_std,
                    y_mean - y_std,
                    color = _color,
                    alpha = 0.05)
########################################################################
handles,labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
ax.set(xlabel = 'Time Steps',
       xticklabels = [f'T-{7-ii}' for ii in range(7)],
       ylabel = 'Feature importance',
       ylim = (-0.05,0.15,))
ax.legend(handles = handles[-2:],labels = labels[-2:],
          title = '',
          loc = 'upper left')
fig.savefig(os.path.join(figure_dir,
                         'features.jpg'),
#            dpi = 400,
#            bbox_inches = 'tight',
            )

unique_experiment = pd.unique(df_plot['experiment'])
fig,axes = plt.subplots(figsize = (30,36),
                        nrows = 4,
                        ncols = int(unique_experiment.shape[0] / 4),
                        sharey = False,)
for ii,(ax,experiment) in enumerate(zip(axes.flatten(),unique_experiment)):
    df_sub = df_plot[df_plot['experiment'] == experiment].sort_values(['acc'])
    df_sub_plot = df_sub.melt(
                          id_vars = ['fold','sub_name','acc','experiment',],
                          value_vars = ['T-7', 'T-6', 'T-5', 'T-4', 'T-3', 'T-2', 'T-1'],)
    df_sub_stats = df_stat_features[df_stat_features['experiment'] == experiment]
    ax = sns.stripplot(x = 'variable',
                       y = 'value',
                       data = df_sub_plot,
                       ax = ax,
                       hue = 'acc',
                       hue_order = xargs['hue_order'],
                       dodge = True,
                       alpha = 1 / df_sub_plot.shape[0] * 100,)
    # compute a less bias mean
    temp_func = partial(stats.trim_mean,**dict(proportiontocut=0.05))
    ax = sns.pointplot(x = 'variable',
                       y = 'value',
                       data = df_sub_plot,
                       ax = ax,
                       hue = 'acc',
                       hue_order = xargs['hue_order'],
                       dodge = 0.4,
                       palette = 'dark',
                       estimator = temp_func,
                       markers = ['D','d'],#'X','x'],
                       join = False,
                       ci = None,
                       scale = 0.5,
                       )
    ########################################################################
    df_sub_plot['x'] = df_sub_plot['variable'].map({f'T-{7-ii}':ii for ii in range(7)})
    df_sub_plot = df_sub_plot.sort_values(['acc'])
    df_stat_features_sub = df_stat_features[df_stat_features['experiment'] == experiment]
    df_stat_features_sub = df_stat_features_sub.sort_values(['acc'])
    df_stat_slope_sub = df_stat_slope[df_stat_slope['experiment'] == experiment]
    df_stat_slope_sub = df_stat_slope_sub.sort_values(['acc'])
    
    
    for (_condition,df_sub_plot_sub),(_,df_stat_features_sub_sub),(_,df_stat_slope_sub_sub),_color in zip(
                            df_sub_plot.groupby(['acc']),
                            df_stat_features_sub.groupby(['acc']),
                            df_stat_slope_sub.groupby(['acc']),
                            xargs['palette']):
        x_vals = df_sub_plot_sub['x'].values
        slopes = df_stat_slope_sub_sub['slope'].values
        intercepts = df_stat_slope_sub_sub['intercept'].values
        xxx = np.linspace(0,6,1000)
        yyy = np.dot(xxx.reshape(-1,1), slopes.reshape(1,-1)) + intercepts
        
        y_mean = temp_func(yyy,axis = 1)
        y_std = np.std(yyy,axis = 1) / np.sqrt(1000)
        ax.plot(xxx,y_mean,linestyle = '--',color = _color,alpha = 0.7)
        ax.fill_between(xxx,
                        y_mean + y_std,
                        y_mean - y_std,
                        color = _color,
                        alpha = 0.05)
    ########################################################################
    
    ax.set(xlabel = '',
           ylabel = '',
           title = f"{experiment}\nn_sample = {int(x_vals.shape[0])}",
           xticklabels = [],
#           ylim = (-0.23,0.23),
           )
    if ii % 4 == 0:
        ax.set(ylabel = 'Feature importance')
    if ii >=12:
        ax.set(xlabel = 'Time Steps',xticklabels = [f'T-{7-ii}' for ii in range(7)])
    handles,labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
#plt.subplots_adjust(top = 1.1)
fig.legend(handles[-4:],labels[-2:],loc = (0.45,0.035),title = '')
fig.savefig(os.path.join(figure_dir,
                         '_features.jpg'),
#            dpi = 400,
#            bbox_inches = 'tight',
            )


fig,ax = plt.subplots(figsize = (16,16))
_vals = []
for _acc,_df_sub in df_stat_slope.groupby(['acc']):
    slopes = _df_sub['slope'].values
    pvals = resample_ttest(slopes,
                           0.,
                           n_ps = 1,
                           n_permutation = int(1e5),
                           n_jobs = -1,
                           verbose = 1,
                           stat_func = np.median,
                           )
    gc.collect()
    _vals.append([stars(pvals),slopes.max() + 0.001])
df_stat_slope['x'] = 0
ax = sns.violinplot(x = 'x',
                   y = 'slope',
                   data = df_stat_slope,
                   ax = ax,
#                   bw = 1,
                   **xargs,
                   )
#plt.setp(ax.collections,alpha = .3)
for jj,(_val,_jitter) in enumerate(zip(_vals,[-.02,.02])):
    ax.annotate(_val[0],
                xy = (0 + _jitter,_val[1]),
                ha = 'center',
                )
ax.set(xlabel = '',
#       xticklabels = ['Correct trials','Incorrect trials'],
       ylabel = r'$\beta$',
       title = "",
       ylim = (-0.003,0.03),
       )
ax.get_legend().set_title('')
fig.savefig(os.path.join(figure_dir,
                         'slopes.jpg'),
        )

fig,axes = plt.subplots(figsize = (30,36),
                        nrows = 4,
                        ncols = int(unique_experiment.shape[0] / 4),
                        sharey = False,)
for ii,(ax,experiment) in enumerate(zip(axes.flatten(),unique_experiment)):
    df_stat_slope_sub = df_stat_slope[df_stat_slope['experiment'] == experiment]
    df_stat_slope_sub = df_stat_slope_sub.sort_values(['acc'])
    _vals = []
    for _acc,_df_sub in df_stat_slope_sub.groupby(['acc']):
        slopes = _df_sub['slope'].values
        pvals = resample_ttest(slopes,
                               0.,
                               n_ps = 1,
                               n_permutation = int(1e5),
                               n_jobs = -1,
                               verbose = 1,
                               stat_func = np.median,)
        _vals.append([stars(pvals),(slopes.max() + slopes.min()) / 2])
    ax = sns.stripplot(x = 'acc',
                       y = 'slope',
                       data = df_stat_slope_sub,
                       ax = ax,
                       marker = '.',
                       dodge = True,
                       )
    for jj,_val in enumerate(_vals):
        ax.annotate(_val[0],
                    xy = (jj,_val[1]),
                    ha = 'center',)
    
    ax.set(xlabel = '',
           ylabel = '',
           title = f"{experiment}",
           xticklabels = [],
#           ylim = (-0.23,0.23),
           )
    if ii % 4 == 0:
        ax.set(ylabel = r'$\beta$')
    if ii >=12:
        ax.set(xlabel = '',xticklabels = ['Correct trials','Incorrect trials'])
fig.tight_layout()
fig.savefig(os.path.join(figure_dir,
                         '_slopes.jpg'),
        )




















