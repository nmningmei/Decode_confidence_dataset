# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:24:08 2020

@author: ning
"""

import os
from glob import glob

from utils import preprocess

for experiment,target_column in zip(['confidence','accuracy','confidence-accuracy'],
                                    [['Confidence'],['accuracy'],['Confidence','accuracy']],):
    data_dir            = '../data'
    model_dir           = '../models/{experiment}'
    working_dir         = '../data/cross_domain'
    working_data        = glob(os.path.join(working_dir,'*', "*.csv"))
    working_df_name     = os.path.join(data_dir,f'{experiment}','cross_domain','all_data.csv')
    saving_dir          = '../results/{experiment}'
    batch_size          = 32
    time_steps          = 7# if experiment != 'confidence-accuracy' else 14
    confidence_range    = 4
    target_columns      = target_column
    n_jobs              = -1
    verbose             = 1
    # make a domain table
    domains             = {item.split('/')[-1]:item.split('/')[-2] for item in working_data}
    df_def              = preprocess(working_data,target_columns = target_columns,n_jobs = n_jobs)
    df_def['temp']      = df_def['filename'].apply(lambda x: x.split('/')[-1])
    df_def['domain']    = df_def['temp'].map(domains)
    if not os.path.exists(os.path.join(data_dir,f'{experiment}','cross_domain')):
        os.makedirs(os.path.join(data_dir,f'{experiment}','cross_domain'))
    df_def.to_csv(working_df_name,index = False)