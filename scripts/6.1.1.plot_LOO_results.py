#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:47:34 2021

@author: nmei
"""
import os

import numpy   as np
import pandas  as pd
import seaborn as sns

from glob import glob

from matplotlib import pyplot as plt

sns.set_style('white')
sns.set_context('paper',font_scale = 2,)

def get_model_type(x):
    if 'C' in x:
        return 'Support Vector Machine'
    elif 'bootstrap' in x:
        return 'Random Forest'
    else:
        return 'Recurrent Neural Network'




if __name__ == "__main__":
    working_dir = '../results/{}/LOO'
    
    x_order = ['confidence','accuracy','confidence-accuracy']
    
    model_names = ['Support Vector Machine',
                   'Random Forest',
                   ]
    
    dfs = {}
    for folder_name in ['confidence','accuracy','confidence-accuracy']:
        working_data = glob(os.path.join(working_dir.format(folder_name),'*.csv'))
        
        # load the data
        df = pd.concat([pd.read_csv(f) for f in working_data])
        df['model_name'] = df['best_params'].apply(get_model_type)
        dfs[folder_name] = df
    
    df_scores = pd.concat([df[['score','r2','feature_type','model_name']] for df in dfs.values()])
    fig,ax = plt.subplots(figsize = (8,5))
    ax = sns.barplot(x = 'feature_type',
                     order = x_order,
                     y = 'score',
                     hue = 'model_name',
                     hue_order = model_names,
                     data = df_scores,
                     seed = 12345,
                     capsize = .1,
                     ax = ax,
                     )
    
    fig,axes = plt.subplots(figsize = (12,5*3),
                            nrows = 3,
                            sharex = True,
                            sharey = True,
                            )
    ax = axes.flatten()[0]
    df_plot = pd.melt(dfs[x_order[0]],
                      id_vars = ['fold','sub_name','feature_type','model_name'],
                      value_vars = [f'features T-{7-ii}' for ii in range(7)],)
    ax = sns.lineplot(x = 'variable',
                      y = 'value',
                      hue = 'model_name',
                      data = df_plot,
                      seed = 12345,
                      sort = False,
                      ax = ax,
                      )
    
    ax = axes.flatten()[1]
    df_plot = pd.melt(dfs[x_order[1]],
                      id_vars = ['fold','sub_name','feature_type','model_name'],
                      value_vars = [f'features T-{7-ii}' for ii in range(7)],)
    ax = sns.lineplot(x = 'variable',
                      y = 'value',
                      hue = 'model_name',
                      data = df_plot,
                      seed = 12345,
                      sort = False,
                      ax = ax,
                      )
    
    ax = axes.flatten()[2]
    df1 = dfs[x_order[2]][['fold', 'score', 'r2', 'n_sample', 'source', 'sub_name', 'best_params',
                           'feature_type', 'features T-14', 'features T-13', 'features T-12',
                           'features T-11', 'features T-10', 'features T-9', 'features T-8',
                           'model_name']]
    df1.columns = ['fold', 'score', 'r2', 'n_sample', 'source', 'sub_name', 'best_params',
                   'feature_type', 'features T-7', 'features T-6', 'features T-5',
                   'features T-4', 'features T-3', 'features T-2', 'features T-1',
                   'model_name']
    df1['feature_type'] = 'confidence'
    df2 = dfs[x_order[2]][['fold', 'score', 'r2', 'n_sample', 'source', 'sub_name', 'best_params',
                           'feature_type', 'features T-7', 'features T-6', 'features T-5',
                           'features T-4', 'features T-3', 'features T-2', 'features T-1',
                           'model_name']]
    df2['feature_type'] = 'accuracy'
    df_plot = pd.concat([df1,df2])
    df_plot = pd.melt(df_plot,
                      id_vars = ['fold','sub_name','feature_type','model_name'],
                      value_vars = [f'features T-{7-ii}' for ii in range(7)],)
    ax = sns.lineplot(x = 'variable',
                      y = 'value',
                      hue = 'model_name',
                      style = 'feature_type',
                      data = df_plot,
                      seed = 12345,
                      sort = False,
                      ax = ax,
                      )