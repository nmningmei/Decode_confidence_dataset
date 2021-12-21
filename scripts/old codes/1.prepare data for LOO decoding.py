#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:20:10 2020

@author: nmei
"""

import os
from glob import glob

from utils import preprocess,get_domains_maps

experiment              = 'confidence'
target_column           = ['Confidence']
data_dir                = '../data'
model_dir               = '../models/{experiment}'
for working_dir in os.listdir('../data/datasets'):
    working_data        = glob(os.path.join('../data/datasets',working_dir, "*.csv"))
    working_df_name     = os.path.join(data_dir,f'{experiment}',f'{get_domains_maps()[working_dir]}.csv')
    time_steps          = 7# if experiment != 'confidence-accuracy' else 14
    confidence_range    = 4
    target_columns      = target_column
    n_jobs              = -1
    verbose             = 1
    
    df_def              = preprocess(working_data,target_columns = target_columns,n_jobs = n_jobs)
    if not os.path.exists(os.path.join(data_dir,f'{experiment}','LOO')):
        os.makedirs(os.path.join(data_dir,f'{experiment}','LOO'))
    df_def.to_csv(working_df_name,index = False)
