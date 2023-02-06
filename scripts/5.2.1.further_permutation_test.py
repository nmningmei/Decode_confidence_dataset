#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:46:58 2023

@author: nmei
"""

import os
import numpy as np
import pandas as pd
from glob import glob

if __name__ == "__main__":
    df = dict(condition = [],
              feature_space = [],
              within_cross = [],
              filename = [],
              score = [],
              model_name = [],
              source = [],
              target = [],
              )
    for condition in ['classification','regression']:
        filenames = glob(os.path.join(f'../results/{condition}/*/*/*csv'))
        for filename in filenames:
            df_temp = pd.read_csv(filename).dropna(axis = 0)
            df_temp['score'] = pd.to_numeric(df_temp['score'])
            filename = filename.replace('cross_domain','cross-domain')
            temp = filename.split('/')
            asdf
            df['condition'].append(condition)
            df['feature_space'].append(temp[3])
            df['within_cross'].append(temp[2])
            df['filename'].append(temp[-1])
            df['score'].append(df_temp['score'].mean())
            df['model_name'].append(temp[-1].split('_')[1])
            df['source'].append(temp[-1].split('_')[-2])
            df['target'].append(temp[-1].split('_')[-1].split('.')[0])
    df = pd.DataFrame(df)