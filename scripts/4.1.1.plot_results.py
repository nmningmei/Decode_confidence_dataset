#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 12:54:01 2021

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
    if 'LinearSV' in x:
        return 'Support Vector Machine'
    elif 'randomforest' in x:
        return 'Random Forest'
    else:
        return 'Recurrent Neural Network'

ylabel_dict = {'regression':'Variance explained',
               'classification':'ROC AUC'}
hline_dict = {'regression':0,
               'classification':0.5}
ylim_dict = {'regression':(-.85,.4),
             'classification':(0,.8)}

if __name__ == "__main__":
    for major_type in ['regression','classification']:
        working_dir = f'../results/{major_type}/replace/*'
        figure_dir = '../figures/'
        
        x_order = ['confidence','accuracy','confidence-accuracy','RT','confidence-RT']
        
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
        g = sns.catplot(x           = 'feature_type',
                        y           = 'score',
                        hue         = 'model_name',
                        hue_order   = model_names,
                        row         = 'target_data',
                        row_order   = CD_order,
                        col         = 'source_data',
                        col_order   = CD_order,
                        data        = df_scores,
                        kind        = 'bar',
                        aspect      = 2,
                        seed        = 12345,
                        capsize     = .1,
                        )
        (g.set(xlabel = '',
               ylabel = ylabel_dict[major_type],
               ylim = ylim_dict[major_type])
          .set_titles("{col_name} --> {row_name}"))
        [ax.axhline(hline_dict[major_type],linestyle = '--',color = 'black',alpha = 0.7) for ax in g.axes.flatten()]
        g.savefig(os.path.join(figure_dir,f'cross {major_type} results.jpg'),
                  dpi = 200,
                  bbox_inches = 'tight')