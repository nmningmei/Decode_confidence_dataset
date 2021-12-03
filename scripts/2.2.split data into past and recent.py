#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 08:09:39 2021

@author: nmei
"""

import os
from glob import glob

import pandas as pd
import numpy as np

working_dir = '../data/'
working_data = glob(os.path.join(working_dir,'*','*','*.csv'))
working_data = [item for item in working_data if ('confidence/' in item) or ('adequacy/' in item)]
working_data = [item for item in working_data if ('past' not in item) and ('recent' not in item)]

for f in working_data:
    df = pd.read_csv(f)
    df_past = df[['filename', 'sub', 'accuracy', 'feature1', 'feature2', 'feature3', 'targets']]
    df_recent = df[['filename', 'sub', 'accuracy', 'feature5', 'feature6', 'feature7', 'targets']]
    df_recent.columns = ['filename', 'sub', 'accuracy', 'feature1', 'feature2', 'feature3', 'targets']
    
    df_past.to_csv(f.replace('.csv','_past.csv'),index = False)
    df_recent.to_csv(f.replace('.csv','_recent.csv'),index = False)