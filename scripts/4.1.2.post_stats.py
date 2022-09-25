#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:05:43 2022

@author: adowa
"""
import os

import numpy   as np
import pandas  as pd

from glob import glob

import utils

def get_model_type(x):
    if 'LinearSV' in x:
        return 'Support Vector Machine'
    elif 'randomforest' in x:
        return 'Random Forest'
    else:
        return 'Recurrent Neural Network'
baseline = dict(regression = 0,
                classification = 0.5)
if __name__ == "__main__":
    for major_type in ['regression','classification']:
        working_dir = f'../results/{major_type}/replace/*'
        figure_dir = '../figures/'
        
        x_order = ['confidence','accuracy','confidence-accuracy','RT','confidence-RT','all']
        
        model_names = ['Support Vector Machine',
                       'Random Forest',
                      #'Recurrent Neural Network',
                       ]
        CD_order = ['Perception','Cognitive','Memory','Mixed']
        
        dfs = {}
        for folder_name in x_order:
            working_data = glob(os.path.join(working_dir.replace('replace',folder_name),'*.csv'))
            
            # load the data
            temp = []
            for f in working_data:
                df = pd.read_csv(f).dropna()
                df['model_name'] = df['best_params'].apply(get_model_type)
                col_names = [item for item in df.columns if ('features' in item)]
                if '_RNN_' in f:
                    for col_name in col_names:
                        df[col_name] = df[col_name].apply(lambda x: -x)
                temp.append(df)
            dfs[folder_name] = pd.concat(temp)
            
        df_scores = pd.concat([df[['score','feature_type','model_name',
                                   'source_data','target_data']] for df in dfs.values()])
        df_scores['score'] = pd.to_numeric(df_scores['score'])
        
        df_res = dict(feature_type = [],
                      target_data = [],
                      source_data = [],
                      model_name = [],
                      score_mean = [],
                      score_std = [],
                      pval = [],
                      )
        for (feature_type,
             target_data,
             source_data,
             model_name),df_sub in df_scores.groupby(['feature_type',
                                                      'target_data',
                                                      'source_data',
                                                      'model_name']):
            np.random.seed(12345)
            pval = utils.resample_ttest(df_sub['score'].values,
                                        baseline = baseline[major_type],
                                        n_permutation = int(1e4),
                                        n_ps = 1,
                                        n_jobs = -1,
                                        verbose = 1,
                                        one_tail = True,
                                        )
            df_res['feature_type'].append(feature_type)
            df_res['target_data'].append(target_data)
            df_res['source_data'].append(source_data)
            df_res['model_name'].append(model_name)
            df_res['score_mean'].append(np.mean(df_sub['score'].values))
            df_res['score_std'].append(np.std(df_sub['score'].values))
            df_res['pval'].append(pval)
        df_res = pd.DataFrame(df_res)
        df_res = df_res.sort_values(['pval'])
        converter = utils.MCPConverter(df_res['pval'].values)
        d = converter.adjust_many()
        df_res['p_corrected'] = d['bonferroni'].values
        df_res['stars'] = df_res['p_corrected'].apply(utils.stars)
        df_res.to_csv(f'../stats/scores_{major_type}.csv',index = False)
        
