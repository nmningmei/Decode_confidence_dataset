#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:03:09 2019

@author: nmei
"""

import os
from glob import glob

import pandas as pd
import numpy as np

working_dir = '../data'
working_data = glob(os.path.join(working_dir,'*.csv'))

for f in working_data:
    df_temp = pd.read_csv(f)
    print(f,[i for i in df_temp.columns if df_temp[i].isnull().any()])
    