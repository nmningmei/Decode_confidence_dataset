#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:55:11 2021

@author: ning
"""

import os
import utils
import pandas as pd
import numpy as np

from glob import glob

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
    df_rf = df_ave[df_ave['decoder'] == 'RF']
    factors = utils.get_groupby_average()[data_type][cv_type]
    factors.remove('decoder')
    df_plot = pd.melt(df_rf,
                      id_vars = factors,
                      value_vars = [f'feature importance T-{7-ii}' for ii in range(time_steps)],
                      var_name = 'Time',
                      value_name = 'feature importance',
                      )
    groupby = list(df_plot.columns)[1:-1]
    results = {name:[] for name in groupby}
    results['ps_mean'] = []
    results['diff_mean'] = []
    for _factors,df_sub in df_plot.groupby(groupby):
        x = df_sub['feature importance'].values
        ps = utils.resample_ttest(x,
                                  baseline = 0,
                                  n_ps = 10,
                                  n_permutation = n_permutation,
                                  one_tail = True,
                                  n_jobs = -1,
                                  verbose = 1,)
        for name,_name in zip(groupby,_factors):
            results[name].append(_name)
        results['ps_mean'].append(np.mean(ps))
        results['diff_mean'].append(np.mean(x))
    results = pd.DataFrame(results)
    results = results.sort_values('ps_mean')
    ps = results['ps_mean'].values
    converter = utils.MCPConverter(pvals = ps)
    d = converter.adjust_many()
    results['ps_corrected'] = d['bonferroni'].values
    if cv_type == 'cross_domain':
        temp = []
        for _,df_sub in results.groupby('source'):
            df_sub = df_sub.sort_values('ps_mean')
            ps = df_sub['ps_mean'].values
            converter = utils.MCPConverter(pvals = ps)
            d = converter.adjust_many()
            df_sub['ps_corrected'] = d['bonferroni'].values
            temp.append(df_sub)
        results = pd.concat(temp)
    results = results.sort_values(list(results.columns[:-3]))
    results['star'] = results['ps_corrected'].apply(utils.stars)
    results.to_csv(os.path.join('../stats',
                                data_type,
                                cv_type,
                                'feature_importance_secondary.csv'),
                   index = False)