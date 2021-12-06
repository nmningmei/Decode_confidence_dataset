#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 07:29:03 2021

@author: nmei
"""

import os
import gc
import utils
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'confidence' # confidence or adequacy
cv_type     = 'cross_domain' # LOO or cross_domain
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
ylim            = (0.35,.85)
star_h          = 0.68
confidence_range= 4
time_steps      = 3
n_permu         = int(1e5)
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


df_ave = utils.load_results(data_type=experiment,
                            within_cross=cv_type,
                            working_data=working_data,)
df_ave['source'] = df_ave['source'].map(dict_source)
factors = ['decoder','accuracy_train','accuracy_test','source']
results = {name:[] for name in factors}
results['ps_mean'] = []
results['diff_mean'] = []
for _factors,df_sub in df_ave.groupby(factors):
    past = df_sub[df_sub['condition'] == 'T-7,T-6,T-5'].sort_values(['fold'])
    recent = df_sub[df_sub['condition'] == 'T-3,T-2,T-1'].sort_values(['fold'])
    
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
temp = []
for _,df_sub in results.groupby(['source']):
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
results = pd.concat(temp)
results['stars'] = results['ps_corrected'].apply(utils.stars)
results = results.sort_values(factors)
results.to_csv(os.path.join(stats_dir,'scores_paired.csv'),index = False)

one_sample = pd.read_csv(os.path.join(stats_dir,'scores_split.csv'))

# plot scores
df_ave['col'] = df_ave['accuracy_test'] + '_' + df_ave['source']
g = sns.catplot(x           = 'decoder',
                y           = 'score',
                col         = 'col',
                data        = df_ave,
                kind        = 'violin',
                aspect      = 1.5,
                hight       = 6,
                **xargs)
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
(g.set_axis_labels("","ROC AUC")
   .set_titles("{row_name} -> {col_name}")
  .set(ylim = ylim)
  )
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
g._legend.set_title("")
## add stars
for ((row,col),df_sub),ax in zip(df_ave.groupby(['accuracy_train','col']),g.axes.flatten()):
    print(row,col,ax.title)
    new_title = ax.get_title().split('_')[0]
    target_data = ax.get_title().split('_')[1]
    ax.set(title=new_title)
    results_sub = one_sample[np.logical_and(one_sample['accuracy_train'] == row,
                                            one_sample['accuracy_test'] == col.split('_')[0])]
    results_sub = results_sub[results_sub['target_data'] == target_data]
    pair_sub = results[np.logical_and(results['accuracy_train'] == row,
                                      results['accuracy_test'] == col.split('_')[0])]
    pair_sub = pair_sub[pair_sub['source'] == target_data]
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
        pair_sub_sub = pair_sub[pair_sub['decoder'] == xtick_label]
        if "*" in pair_sub_sub['stars'].values[0]:
            ax.plot([ii-adjustment,ii-adjustment,
                     ii+adjustment,ii+adjustment],
                    [0.72,0.74,0.74,0.72],
                    color = 'black')
            ax.annotate(pair_sub_sub['stars'].values[0],
                        xy = (ii,0.75),
                        ha = 'center',
                        va = 'center',
                        fontsize = 14,
                        )

plt.suptitle(f'{" "*100}'.join(
    domains),x = 0.4,y = 1.15)
plt.subplots_adjust(top = 1.)

g.savefig(os.path.join(figures_dir,'scores_split.jpg'),
          dpi = 300,
          bbox_inches = 'tight')









