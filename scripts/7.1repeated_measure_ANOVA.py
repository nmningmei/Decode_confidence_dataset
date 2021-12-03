#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:10:52 2021

@author: ning
"""

import os
import utils

from glob import glob

working_dir = '../results'
data_types = ['confidence','adequacy']
_within_cross = ['LOO','cross_domain']
dict_rename     = {0:'incorrect trials',1:'correct trials'}
dict_condition  = {'past':'T-7,T-6,T-5','recent':'T-3,T-2,T-1'}
subject_factor = {"LOO":"study_name",
                  "cross_domain":"filename"}

for data_type in data_types:
    for within_cross in _within_cross:
        working_data = glob(os.path.join(working_dir,
                                         data_type,
                                         within_cross,
                                         '*.csv'))
        working_data = [item for item in working_data if ('past' in item) \
                                                      or ('recent' in item)]
        df_ave = utils.load_results(data_type = data_type,
                                    within_cross = within_cross,
                                    working_data = working_data,
                                    dict_rename = dict_rename,
                                    dict_condition = dict_condition,
                                    )
        print(data_type,within_cross,df_ave.shape,df_ave.columns)
        df_ave.to_csv(f'{working_dir}/for_anova/{data_type}_{within_cross}.csv',index = False)
