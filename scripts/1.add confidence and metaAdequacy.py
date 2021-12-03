#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:20:10 2020

@author: nmei
"""

import os
from glob import glob

from utils import preprocess

for experiment,target_column in zip(['confidence','adequacy'],['Confidence','metaAdequacy']):
    data_dir            = '../data'
    model_dir           = '../models/{experiment}'
    working_dir         = '../data/4-point'
    working_data        = glob(os.path.join(working_dir, "*.csv"))
    working_df_name     = os.path.join(data_dir,f'{experiment}','LOO','all_data.csv')
    saving_dir          = '../results/{experiment}'
    batch_size          = 32
    time_steps          = 7
    confidence_range    = 4
    target_columns      = [target_column]
    n_jobs              = -1
    verbose             = 1
    
    df_def          = preprocess(working_data,target_columns = target_columns,n_jobs = n_jobs)
    if not os.path.exists(os.path.join(data_dir,f'{experiment}')):
        os.makedirs(os.path.join(data_dir,f'{experiment}','LOO'))
    df_def.to_csv(working_df_name,index=False)
