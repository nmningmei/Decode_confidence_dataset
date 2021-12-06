#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 06:44:21 2021

@author: nmei
"""

import os
import gc
import utils
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'confidence' # confidence or adequacy
cv_type     = 'LOO' # LOO or cross_domain
n_permu     = int(1e5)
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
ylim    = {'confidence':(0.49,1.0),
           'adequacy':(0.3,0.9)}[experiment]
confidence_range= 4
time_steps      = 3 #
dict_rename     = {0:'incorrect trials',1:'correct trials'}
dict_condition  = {'past':'T-7,T-6,T-5','recent':'T-3,T-2,T-1'}
xargs           = dict(split        = True,
                       inner        = 'quartile',
                       cut          = 0,
                       scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       row_order    = ['correct trials','incorrect trials',],
                       col_order    = ['SVM','RF','RNN'],
                       hue_order    = list(dict_condition.values()),
                       )
df_ave = utils.load_results(data_type=experiment,
                            within_cross=cv_type,
                            working_data=working_data,)
factors = ['decoder','accuracy_train','accuracy_test',]
results = {name:[] for name in factors}
results['ps_mean'] = []
results['diff_mean'] = []
for _factors,df_sub in df_ave.groupby(factors):
    past = df_sub[df_sub['condition'] == 'T-7,T-6,T-5'].sort_values(['study_name'])
    recent = df_sub[df_sub['condition'] == 'T-3,T-2,T-1'].sort_values(['study_name'])
    
    a = past['score'].values
    b = recent['score'].values
    gc.collect()
    ps = utils.resample_ttest_2sample(b,a,
                                      n_ps = 10,
                                      n_permutation = n_permu,
                                      match_sample_size = True,
                                      one_tail = True,
                                      n_jobs = -1,
                                      )
    gc.collect()
    
    [results[name].append(_name) for name,_name in zip(factors,_factors)]
    results['ps_mean'].append(np.mean(ps))
    results['diff_mean'].append(np.abs(a.mean() - b.mean()))
results = pd.DataFrame(results)
results = results.sort_values(['ps_mean'])
converter = utils.MCPConverter(pvals = results['ps_mean'].values)
d = converter.adjust_many()
results['ps_corrected'] = d['bonferroni'].values
results['stars'] = results['ps_corrected'].apply(utils.stars)
results = results.sort_values(factors)
results.to_csv(os.path.join(stats_dir,'scores_paired.csv'),index=False)

one_sample = pd.read_csv(os.path.join(stats_dir,'scores_split.csv'))

# plot scores
g = sns.catplot(x           = 'accuracy_train',
                y           = 'score',
                col         = 'decoder',
                hue         = 'condition',
                row         = 'accuracy_test',
                data        = df_ave,
                kind        = 'violin',
                aspect      = 1.5,
                **xargs)
(g.set_axis_labels("Training data","ROC AUC")
  .set_titles("Tested on {row_name} | {col_name}")
  .set(ylim = ylim))
# [ax.set_title(title) for ax,title in zip(g.axes.flatten(),['Linear support vector machine','Random forest','Recurrent neural network'])]
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
g._legend.set_title("")
## add stars
for axes,_accuracy_test in zip(g.axes,xargs['row_order']):
    for ax,_decoder in zip(axes,xargs['col_order']):
        df_sub = one_sample[np.logical_and(one_sample['accuracy_test'] == _accuracy_test,
                                        one_sample['decoder'] == _decoder)]
        df_sub = df_sub.sort_values(['accuracy_train','accuracy_test','condition'])
        xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
        
        for ii,text_obj in enumerate(xtick_order):
            position        = text_obj.get_position()
            xtick_label     = text_obj.get_text()
            df_sub_stats    = df_sub[df_sub['accuracy_train'] == xtick_label].sort_values(['accuracy_test'])
            for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.125,0.125]):
                if '*' in temp_row['stars']:
                    print(temp_row['stars'])
                    ax.annotate(temp_row['stars'],
                                xy          = (ii + adjustment,.85),
                                ha          = 'center',
                                fontsize    = 14)
## add pair stars
adjustment = 0.125
for axes,_accuracy_test in zip(g.axes,xargs['row_order']):
    for ax,_decoder in zip(axes,xargs['col_order']):
        df_sub = results[np.logical_and(results['accuracy_test'] == _accuracy_test,
                                        results['decoder'] == _decoder)]
        df_sub = df_sub.sort_values(['accuracy_train','accuracy_test'])
        xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
        
        for ii,text_obj in enumerate(xtick_order):
            position        = text_obj.get_position()
            xtick_label     = text_obj.get_text()
            df_sub_stats    = df_sub[df_sub['accuracy_train'] == xtick_label].sort_values(['accuracy_test'])
            if "*" in df_sub_stats['stars'].values[0]:
                ax.plot([ii-adjustment,ii-adjustment,
                         ii+adjustment,ii+adjustment],
                        [0.88,0.9,0.9,0.88],
                        color = 'black')
                ax.annotate(df_sub_stats['stars'].values[0].replace('*','o'),
                            xy = (ii,0.92),
                            ha = 'center',
                            va = 'center',
                            fontsize = 14,
                            )

g.savefig(os.path.join(figures_dir,'scores_split.jpg'),
          dpi = 300,
          bbox_inches = 'tight')