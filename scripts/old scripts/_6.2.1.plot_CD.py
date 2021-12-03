#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:16:06 2019

@author: nmei
"""

import os
from glob import glob
from sklearn.preprocessing import MinMaxScaler as scaler
import pandas as pd
import numpy as np
from scipy import stats
from functools import partial
from utils import resample_ttest,stars,get_array_from_dataframe
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

experiment = 'confidence' # confidence or adequacy
_decoder = 'regression'
working_dir = f'../results/{experiment}/cross_domain/'
stats_dir = f'../stats/{experiment}/CD'
figure_dir = f'../figures/{experiment}/CD'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
working_data = np.sort(glob(os.path.join(working_dir,"*.csv")))

df = []
for f in working_data:
    temp = pd.read_csv(f)
    decoder = f.split('/')[-1].split(' ')[0].split('_')[0]
    experiment = f.split('/')[-1].split('(')[-1].split(')')[0]
    temp['model'] = decoder
    temp['experiment'] = experiment
    col_to_rename = [item for item in temp.columns if ('T-' in item)]
    rename_mapper = {item:f'{item.split(" ")[-1]}' for item in col_to_rename}
    temp = temp.rename(columns = rename_mapper)
    df.append(temp)
df = pd.concat(df)

df_plot = df[np.logical_and(df['source'] != 'train',df['model'] == _decoder)]

# further process the data for plotting
df_plot['acc_train']  = df_plot['accuracy_train'].map({0:'incorrect',1:'correct'})
df_plot['acc_test'] = df_plot['accuracy_test'].map({0:'incorrect',1:'correct'})
df_plot['condition'] = df_plot['acc_train'] + '->' + df_plot['acc_test']
df_plot = df_plot.sort_values(['experiment','model','condition'])

df_stat_score = pd.read_csv(os.path.join(stats_dir,'scores.csv'))
df_stat_score = df_stat_score[df_stat_score['model'] == _decoder]

df_stat_slope = pd.read_csv(os.path.join(stats_dir,'slopes.csv'))
df_stat_slope = df_stat_slope[df_stat_slope['model'] == _decoder]

df_stat_features = pd.read_csv(os.path.join(stats_dir,'features.csv'))
df_stat_features = df_stat_features[df_stat_features['model'] == _decoder]

xargs = dict(hue = 'acc_test',
#             hue_order = ['RF_correct','RF_incorrect','RNN_correct','RNN_incorrect'],
             hue_order = ['correct','incorrect',],
             split = True,
             inner = 'quartile',
             cut = 0,
             scale = 'width',
             palette = ['deepskyblue','tomato'])


fig,axes = plt.subplots(figsize = (16*2,16),
                        ncols = 2)
for (acc_train,df_plot_sub),(_,df_stat_score_sub),ax in zip(df_plot.groupby('acc_train'),
                                                            df_stat_score.groupby('acc_train'),
                                                            axes.flatten()):
    ax = sns.violinplot(x = 'source',
                        y = 'score',
                        data = df_plot_sub,
                        ax = ax,
                        **xargs,
                        )
    handles,labels = ax.get_legend_handles_labels()
    xtick_order = list(ax.xaxis.get_majorticklabels())
    
    for ii,text_obj in enumerate(xtick_order):
        position = text_obj.get_position()
        ytick_label = text_obj.get_text()
        df_sub_stats = df_stat_score_sub[df_stat_score_sub['source'] == ytick_label].sort_values(['acc_train'])
        for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.125,0.125]):
            if '*' in temp_row['stars']:
                print(temp_row['stars'])
                ax.annotate(temp_row['stars'],
                            xy = (ii + adjustment,0.95),
                            ha = 'center',
                            fontsize = 14)
    ax.set(ylim = (0.15,1.05),ylabel = 'ROC AUC',xlabel = 'Target study',title = f'Trained on {acc_train} trials')
    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
    ax.legend(handles = handles,labels = [f'{item} trials' for item in labels],loc = 'lower right',title = 'Test data')

# fig.savefig(os.path.join(figure_dir,'scores.jpg',
#                          ),
#             bbox_inches = 'tight',)

_df_plot = pd.melt(df_plot,id_vars = [item for item in df_plot.columns if ('T' not in item)],
                  value_vars = [item for item in df_plot.columns if ('T' in item)],
                  var_name = ['Time'],
                  value_name = 'Weights')

fig,axes = plt.subplots(figsize = (16,7 * 3),
                        # ncols = 2,
                        nrows = 3,
                        sharey = False,
                        )
for ax,((source),df_sub_plot) in zip(axes.flatten(),_df_plot.groupby(['source'])):
    # ax = sns.violinplot(x = 'Time',
    #                  y = 'Weights',
    #                  data = df_sub_plot,
    #                  ax = ax,
    #                  **xargs)
    temp_func = partial(stats.trim_mean,**dict(proportiontocut=0.05))
    ax = sns.pointplot(x = 'Time',
                        y = 'Weights',
                        data = df_sub_plot,
                        ax = ax,
                        hue = 'acc_train',
                        hue_order = xargs['hue_order'],
                        dodge = 0.4,
                        palette = 'dark',
                        estimator = temp_func,
                        markers = ['D','d'],#'X','x'],
                        join = False,
                        ci = 95,
                        scale = 0.75,
                        )
    
    ###########################################################################
    df_sub_plot['x'] = df_sub_plot['Time'].map({f'T-{7-ii}':ii for ii in range(7)})
    df_sub_plot = df_sub_plot.sort_values(['acc_test'])
    df_stat_features_sub = df_stat_features[#np.logical_and(
                            df_stat_features['source'] == source#,
                            # df_stat_features['acc_train'] == acc_train
                            # )
        ]
    df_stat_features_sub = df_stat_features_sub.sort_values(['acc_test'])
    df_stat_slope_sub = df_stat_slope[#np.logical_and(
                            df_stat_slope['source'] == source#,
                            # df_stat_slope['acc_train'] == acc_train
                            # )
        ]
    df_stat_slope_sub = df_stat_slope_sub.sort_values(['acc_test'])
    
    for (_condition,df_sub_plot_sub),(_,df_stat_features_sub_sub),(_,df_stat_slope_sub_sub),_color in zip(
                            df_sub_plot.groupby(['acc_train']),
                            df_stat_features_sub.groupby(['acc_train']),
                            df_stat_slope_sub.groupby(['acc_train']),
                            xargs['palette']):
        x_vals = df_sub_plot_sub['x'].values
        xxx = np.linspace(0,6,1000)
        slopes = df_stat_slope_sub_sub['slope'].values
        intercepts = df_stat_slope_sub_sub['intercept'].values
        yyy = np.dot(xxx.reshape(-1,1), slopes.reshape(1,-1)) + intercepts
        
#        y_mean = get_array_from_dataframe(df_stat_features_sub_sub,'y_mean')
#        y_std = get_array_from_dataframe(df_stat_features_sub_sub,'y_std')
        y_mean = yyy.mean(1)
        y_std = yyy.std(1)
        
        ax.plot(xxx,y_mean,linestyle = '--',color = _color,alpha = 0.7)
        ax.fill_between(xxx,
                        y_mean + y_std,
                        y_mean - y_std,
                        color = _color,
                        alpha = 0.2)
        
    ###########################################################################
    ax.set(xlabel = '',
           ylabel = 'Feature importance',
           title = f'{source},',
           xticklabels = [],
           # ylim = (-0.1,0.2,),
           )
    handles,labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
ax.set(xlabel = 'Time Steps',xticklabels = [f'T-{7-ii}' for ii in range(7)])
fig.legend(handles[-2:],
           labels[-2:],
           loc = (0.12,0.55),
           title = '',
           borderaxespad = 1.)
# fig.savefig(os.path.join(figure_dir,'features.jpg'),
#           bbox_inches = 'tight')
asfd
fig,axes = plt.subplots(figsize = (12,6 * 3),
                        nrows = 3,
                        sharey = False,)
for ii,((source,df_stat_slope_sub),ax) in enumerate(zip(
                                        df_stat_slope.groupby(['source']),
                                        axes.flatten())):
    ax = sns.stripplot(x = 'acc_train',
                       y = 'slope',
                       data = df_stat_slope_sub,
                       ax = ax,
                       order = xargs['hue_order'],
                       palette = xargs['palette'],
                       )
    ax.set(xlabel = '',
           ylabel = r'$\beta$',
           xticklabels = ['Correct trials','Incorrect trials'],
           title = f'Target study: {source}',
           )
fig.tight_layout()
fig.savefig(os.path.join(figure_dir,'slopes.jpg'),
            bbox_inches = 'tight',)

