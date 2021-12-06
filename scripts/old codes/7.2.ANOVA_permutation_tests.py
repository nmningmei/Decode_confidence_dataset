#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 04:42:05 2021

@author: nmei
"""

import os
import utils
import pandas as pd
import numpy as np

from glob import glob

# paired comparison
data_type = 'confidence'
confidence_range = 4
time_steps = 7
n_permutation = int(1e4)
dict_rename = {0:'incorrect trials',1:'correct trials'}
working_dir = f'../results/{data_type}'
main_effect = dict(LOO = 'accuracy_train',
                   cross_domain = 'accuracy_train')
res = dict()
for cv_type in ['LOO','cross_domain']:
    working_data = glob(os.path.join(working_dir,cv_type,'*.csv'))
    working_data = [item for item in working_data if\
                    ('past' not in item) and ('recent' not in item)]
    df_ave = utils.load_results(
        data_type      = data_type,
        within_cross   = cv_type,
        working_data   = working_data,
        dict_rename    = dict_rename,
        dict_condition = None,
        )
    df_ave.to_csv(f'../results/for_anova/scores_{cv_type}.csv',
                  index = False)
    # perform ANOVA on the data using R
    factors = utils.get_groupby_average()[data_type][cv_type]
    factors.remove(main_effect[cv_type])
    results = {name:[] for name in factors[1:]}
    results['ps_mean'] = []
    results['diff_mean'] = []
    for _factors,df_sub in df_ave.groupby(factors[1:]):
        correct = df_sub[df_sub[main_effect[cv_type]] == 'correct trials']
        incorrect = df_sub[df_sub[main_effect[cv_type]] == 'incorrect trials']
        
        # sort the rows
        correct = correct.sort_values(factors[1])
        incorrect = incorrect.sort_values(factors[1])
        
        ps = utils.resample_ttest_2sample(correct['score'].values,
                                          incorrect['score'].values,
                                          n_ps = 10,
                                          n_permutation = n_permutation,
                                          one_tail = True,
                                          match_sample_size = True,
                                          n_jobs = -1,
                                          verbose = 1,
                                          )
        for name,_name in zip(factors[1:],_factors):
            results[name].append(_name)
        results['ps_mean'].append(np.mean(ps))
        results['diff_mean'].append(np.abs(correct['score'].values.mean() - incorrect['score'].values.mean()))
    res[cv_type] = pd.DataFrame(results)

for key,df_sub in res.items():
    df_sub = df_sub.sort_values('ps_mean')
    ps = df_sub['ps_mean'].values
    converter = utils.MCPConverter(pvals=ps)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    df_sub = df_sub.sort_values(list(df_sub.columns[:-3]))
    res[key] = df_sub



























