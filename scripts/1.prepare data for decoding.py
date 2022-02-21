#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:20:10 2020

@author: nmei

experiment              | target_column
'confidence'            | ['Confience']
'accuracy'              | ['accuracy']
'confidence-accuracy'   | ['Confidence','accuracy']
'RT'                    | ['RT']
'confidence-RT'         | ['Confidence','RT']

preprocess steps:
    1. concatenate all the study within a domain
    2. keras::TimeseriesGenerator is used to generate features from T-1 to T-7 and the corresponding target
    3. confidence features are checked if they are in between 1 and 4
"""

import os
from glob import glob

from utils import preprocess,get_domains_maps

# experiment              = 'confidence-accuracy'
# target_column           = ['Confidence','accuracy']
for experiment,target_column in zip(['confidence',
                                     'accuracy',
                                     'confidence-accuracy',
                                     'RT',
                                     'confidence-RT',
                                     ],
                                    [['Confidence'],
                                     ['accuracy'],
                                     ['Confidence','accuracy'],
                                     ['RT'],
                                     ['Confidence','RT'],
                                     ]):
    data_dir                = '../data'
    model_dir               = '../models/{experiment}'
    for working_dir in ['mixed_4-point',
                        #'mem_4-point',
                        #'4-point',
                        #'cognitive-4-rating'
                        ]:
        working_data        = glob(os.path.join('../data/datasets',working_dir, "*.csv"))
        working_df_name     = os.path.join(data_dir,f'{experiment}',f'{get_domains_maps()[working_dir]}.csv')
        time_steps          = 7# if experiment != 'confidence-accuracy' else 14
        confidence_range    = 4
        target_columns      = target_column
        n_jobs              = -1
        verbose             = 1
        
        # if 'RT' not in experiment and working_dir != 'cognitive-4-rating':
        df_def              = preprocess(working_data,target_columns = target_columns,n_jobs = n_jobs)
        if not os.path.exists(os.path.join(data_dir,f'{experiment}')):
            os.makedirs(os.path.join(data_dir,f'{experiment}'))
        df_def.to_csv(working_df_name,index = False)
