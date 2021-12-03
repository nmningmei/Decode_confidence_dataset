# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:24:08 2020

@author: ning
"""

import os
from glob import glob
from utils import preprocess
import numpy as np

for name in ['confidence','adequacy']:
    experiment          = [name,'cross_domain']
    data_dir            = '../data/'
    model_dir           = '../models'
    source_dir          = '../data/4-point'
    target_dir          = '../data/targets/*/'
    source_data         = np.array(glob(os.path.join(source_dir, "*.csv")))
    target_data         = np.array(glob(os.path.join(target_dir, "*.csv")))
    source_df_name      = os.path.join(data_dir,f'{experiment[0]}',f'{experiment[1]}','source.csv')
    target_df_name      = os.path.join(data_dir,f'{experiment[0]}',f'{experiment[1]}','target.csv')
    time_steps          = 7
    confidence_range    = 4
    n_jobs              = -1
    target_columns      = {'confidence':'Confidence',
                           'adequacy':'metaAdequacy'}[experiment[0]]
    
    if not os.path.exists(os.path.join(data_dir,f'{experiment[0]}',f'{experiment[1]}',)):
        os.mkdir(os.path.join(data_dir,f'{experiment[0]}',f'{experiment[1]}',))
    
    df_source       = preprocess(source_data,target_columns = [target_columns],n_jobs = n_jobs)
    df_target       = preprocess(target_data,target_columns = [target_columns],n_jobs = n_jobs)
    
    df_source.to_csv(source_df_name,index = False)
    df_target.to_csv(target_df_name,index = False)