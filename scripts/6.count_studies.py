# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:47:15 2021

@author: ning
"""

import os

from glob import glob

import numpy as np
import pandas as pd

working_dir = '../data/confidence/cross_domain'
perception = os.path.join(working_dir,'source.csv')
others = os.path.join(working_dir,'target.csv')

df_perception = pd.read_csv(perception)
df_others = pd.read_csv(others)
df_others['domain'] = df_others['filename'].apply(lambda x:x.split('/')[3].split('-')[0])

temp = df_perception.groupby(['filename','sub']).count().reset_index()[['filename','sub','targets']]
trials = temp.groupby(['filename']).sum().reset_index()['targets'].values
temp = temp.groupby(['filename']).count().reset_index()
temp['trials'] = trials
temp['domain'] = 'Perceptual domain'
df_percept = temp.copy()

# count trials
temp = df_others.groupby(['domain','filename','sub']).count().reset_index()[['domain','filename','sub','targets']]
trials = temp.groupby(['filename']).sum().reset_index()['targets'].values
temp = temp.groupby(['domain','filename',]).count().reset_index()
temp['trials'] = trials

df = pd.concat([df_percept,temp])

df['study'] = df['filename'].apply(lambda x:x.split('/')[-1].replace('.csv',''))

def _process(x):
    temp = x.split('_')
    if len(temp) == 3:
        return f'{temp[1]},{temp[2]}'
    elif len(temp) == 4:
        return f'{temp[1]}, {temp[2]}, {temp[3]}'
df['study'] = df['study'].apply(_process)

df.to_csv('../results/study counts.csv',index = False)


df_perception['domain'] = 'perceptual'
df = pd.concat([df_perception,df_others])

# count proportions
temp = df.groupby(['filename','domain','accuracy']).count().reset_index()[['filename','domain','accuracy','targets']]
counts = temp['targets'].values
temp['proportion'] = temp['targets'].values / np.repeat(counts.reshape(-1,2).sum(1),2)

# count # of subjects
temp = (df.groupby(['sub','domain','accuracy','filename'])
          .count()
          .reset_index()
          .groupby(['filename','domain','accuracy'])
          .count()
          .reset_index()['sub'].sum())
