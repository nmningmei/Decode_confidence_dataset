#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:36:16 2019

@author: ningmei
"""
import re
import gc

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    from tensorflow.keras       import layers, Model, optimizers, losses, regularizers
except:
    pass

from tqdm import tqdm

from joblib import Parallel,delayed

# from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from scipy.special import softmax

gc.collect()

def preprocess(working_data,
               time_steps = 7,
               target_columns = ['Confidence'],
               n_jobs = 8,
               verbose = 1
               ):
    """
    Parameters
    -----------------------
    working_data : list of strings, csv names
    time_steps : int, default = 7
        trials looking back
    target_columns : list of strings, default = ['Confidence']
        columns names that we want to parse to become the RNN features, i.e. Confidence, accuracy, RTs
    n_jobs : int, default = 8
        number of CPUs we want to parallize the for-loop job, use -1 to use all CPUs
    verbose: int or bool, default = 1
        if > 0, we print out the parallized for-loop processes
    
    Returns
    -----------------------
    df_def: pandas.DataFrame,
        concatenated pandas dataframe that contains features at each time point and targets
    """
    df_for_concat = []
    # define iterator
    t = tqdm(working_data,)
    for f in t:
        # print(f)
        df_temp             = pd.read_csv(f,header = 0)
        # some data was not standardized
        if "Siedlecka_2018" in f:
            df_temp = df_temp[df_temp['Session'] != 9]
        if not pd.api.types.is_object_dtype(df_temp['Stimulus']) and pd.api.types.is_object_dtype(df_temp['Response']):
            df_temp['Stimulus'] = df_temp['Stimulus'].apply(str2int)
            df_temp['Response'] = df_temp['Response'].apply(str2int)
        if ("Siedlecka" in f) and ('Exp' in f):
            df_temp['Response'] = df_temp['Response'].values - 1
        if 'Koculak_unpub' in f: # this one is super strange
            df_temp['accuracy'] = df_temp['Accuracy'].values.copy()
        else:
            df_temp['accuracy'] = np.array(df_temp['Stimulus'].values == df_temp['Response'].values,dtype = int)
        df_temp['filename'] = f
        if len(target_columns) == 2:
            df_temp = df_temp[np.concatenate([['Subj_idx','filename',],target_columns])]
        elif 'accuracy' not in target_columns:
            df_temp = df_temp[np.concatenate([['Subj_idx','filename','accuracy'],target_columns])]
        else:
            df_temp = df_temp[['Subj_idx','filename','accuracy']]
        df_for_concat.append(df_temp)
    df_concat = pd.concat(df_for_concat)
    
    # initialize
    df = dict(sub       = [],
              filename  = [],
              targets   = [],
              accuracy  = [],
              )
    if len(target_columns) == 1:
        for ii in range(time_steps):
            df[f'feature{ii + 1}'] = []
    elif len(target_columns) == 2:
        for ii in range(time_steps * 2):
            df[f'feature{ii + 1}'] = []
    
    t = tqdm(df_concat.groupby(['Subj_idx','filename']),)
    for (sub,filename), df_sub in t:
    #    print(sub,filename)
        
        values      = df_sub[target_columns].values
        accuracy    = df_sub['accuracy'].values
        # tensorflow.keras.preprocessing.TimeseriesGenerator
        data_gen    = TimeseriesGenerator(values,
                                          values,
                                          length        = time_steps,
                                          sampling_rate = 1,
                                          batch_size    = 1,
                                          )
        t.set_description(f'{filename} sub-{sub} {np.mean(accuracy[time_steps:]):.2f}')
        for (features_,targets_),accuracy_ in zip(list(data_gen),accuracy[time_steps:]):
            df["sub"        ].append(sub)
            df["filename"   ].append(filename)
            df["targets"    ].append(targets_.flatten()[0])
            df["accuracy"   ].append(accuracy_)
            if len(target_columns) == 1:
                features_ = features_.flatten()
            elif len(target_columns) == 2:
                features_ = np.hstack([features_[0,:,0],features_[0,:,1]])
            [df[f"feature{ii + 1}"].append(f) for ii,f in enumerate(features_)]
            
    df = pd.DataFrame(df)
    # re-order the columns
    df = df[np.concatenate([
             ['filename', 'sub','accuracy'],
             [f'feature{ii + 1}' for ii in range(time_steps * len(target_columns))],
             ['targets']
             ])]
    if 'Confidence' in target_columns:
        """
        REMOVING FEATURES AND TARGETS DIFFERENT FROM 1-4
        """
        df_temp = df.dropna()
        ###################### parallelize the for-loop to multiple CPUs ############################
        ###################### it is faster than df_temp.apply           ############################
        def detect(row):
            col_names   = np.concatenate([[f'feature{ii+1}' for ii in range(time_steps)],['targets']])
            values      = np.array([row[col_name] for col_name in col_names])
            return np.logical_and(values < 5, values > 0)
        
        idx_within_range = Parallel(n_jobs  = n_jobs,
                                    verbose = verbose,
                                    )(delayed(detect)(**{'row':row})for ii,row in df_temp.iterrows())
        #############################################################################################
        # ALL df_pepe columns must be true(1) & sum up to 8
        idx     = np.sum(idx_within_range,axis = 1) == (time_steps + 1)
        df_def  = df_temp.loc[idx,:]
    else:
        df_def = df.dropna()
    return df_def

def str2int(x):
    try:
        x = int(x)
    except:
        x = np.nan
    return x

def meta_adequacy(x):
    """
    If  accuracy is 1 and Confidence is  1 then adequacy is  1 
    If  accuracy is 1 and Confidence is  2 then adequacy is  2 
    If  accuracy is 1 and Confidence is  3 then adequacy is  3 
    If  accuracy is 1 and Confidence is  4 then adequacy is  4 
    If  accuracy is 0 and Confidence is  1 then adequacy is  4
    If  accuracy is 0 and Confidence is  2 then adequacy is  3
    If  accuracy is 0 and Confidence is  3 then adequacy is  2 
    If  accuracy is 0 and Confidence is  4 then adequacy is  1
    """
    if x['accuracy'] == 1:
        return x['Confidence']
    else:
        return 5 - x['Confidence']

def check_column_type(df_sub):
    for name in df_sub.columns:
        if name == 'filename':
            pass
        elif name == 'temp':
            pass
        elif name == 'domain':
            pass
        elif name == 'sub':
            try:
                df_sub[name] = df_sub[name].astype(int)
            except:
                df_sub[name] = df_sub[name].astype(str)
        else:
            df_sub[name] = df_sub[name].astype(int)
    return df_sub

# the most important helper function: early stopping and model saving
def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks             import ModelCheckpoint,EarlyStopping
    """
    Make call back function lists for the keras models
    
    Parameters
    -------------------------
    model_name : str,
        directory of where we want to save the model and its name
    monitor : str, default = 'val_loss'
        the criterion we used for saving or stopping the model
    mode : str, default = 'min'
        min --> lower the better, max --> higher the better
    verboser : int or bool, default = 0
        printout the monitoring messages
    min_delta : float, default = 1e-4
        minimum change for early stopping
    patience : int, default = 50
        temporal windows of the minimum change monitoring
    frequency : int, default = 1
        temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint : tensorflow.keras.callbacks
        saving the best model
    EarlyStopping : tensorflow.keras.callbacks
        early stoppi
    """
    checkPoint = ModelCheckpoint(model_name,# saving path
                                 monitor          = monitor,# saving criterion
                                 save_best_only   = True,# save only the best model
                                 mode             = mode,# saving criterion
                                 verbose          = verbose,# print out (>1) or not (0)
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
                                 )
    return [checkPoint,earlyStop]

def make_hidden_state_dataframe(states,sign = -1,time_steps = 7,):
    df = pd.DataFrame(sign * states[:,:,0],columns = [f'T{ii - time_steps}' for ii in range(time_steps)])
    df = pd.melt(df,value_vars = df.columns,var_name = ['Time'],value_name = 'Hidden Activation')
    return df

def convert_object_to_float(df):
    for name in df.columns:
        if name == 'filename':
            pass
        else:
            try:
                df[name] = df[name].apply(lambda x:int(re.findall('\d+',x)[0]))
            except:
                print(f'column {name} contains strings')
    return df

def build_SVMRegressor(max_iter = int(1e3),):
    from sklearn.svm import LinearSVR
    svm = LinearSVR(random_state = 12345,max_iter = max_iter,)
    return svm

def build_RF(n_jobs             = 1,
             max_depth          = 3,
             n_estimators       = 100,
             oob_score          = True,
             bootstrap          = True):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators     = n_estimators,
                               # criterion        = 'squred_error',
                               max_depth        = max_depth,
                               n_jobs           = n_jobs,
                               bootstrap        = bootstrap,
                               oob_score        = oob_score,
                               random_state     = 12345,
                               )
    return rf

def build_RNN(time_steps = 7,confidence_range = 4,input_dim = 1,model_name = 'temp.h5'):
    # reset the GPU memory
    tf.keras.backend.clear_session()
    try:
        tf.random.set_random_seed(12345) # tf 1.0
    except:
        tf.random.set_seed(12345) # tf 2.0
    # build a 3-layer RNN model
    inputs                  = layers.Input(shape     = (time_steps,input_dim),# time steps by features 
                                           name      = 'inputs')
    # the recurrent layer
    lstm,state_h,state_c    = layers.LSTM(units             = 1,
                                          return_sequences  = True,
                                          return_state      = True,
                                          # kernel_regularizer= regularizers.L1L2(1e-3,1e-3),
                                          name              = "lstm")(inputs)
    # from the LSTM layer, we will have an output with time steps by features, but 
    dimension_squeeze       = layers.Lambda(lambda x:tf.keras.backend.squeeze(x,2))(lstm)
    outputs                 = layers.Dense(1,
                                           # kernel_regularizer= regularizers.L1L2(1e-3,1e-3),
                                           name             = "output",
                                           activation       = "sigmoid")(dimension_squeeze)
    model                   = Model(inputs,
                                    outputs)
    
    model.compile(optimizer     = optimizers.SGD(learning_rate = 1e-2),
                  loss          = losses.mean_absolute_error,
                  metrics       = ['mse'])
    # early stopping
    callbacks = make_CallBackList(model_name    = model_name,
                                  monitor       = 'val_loss',
                                  mode          = 'min',
                                  verbose       = 0,
                                  min_delta     = 1e-4,
                                  patience      = 5,
                                  frequency     = 1,)
    return model,callbacks

def get_domains_maps():
    temp = {'4-point':'Perception', 
            'cognitive-4-rating':'Cognitive', 
            'mem_4-point':'Memory', 
            'mixed_4-point':'Mixed'}
    return temp

def get_feature_targets(df_sub,
                        n_features = 7,
                        time_steps = 7,
                        target_attributes = 'confidence',
                        group_col = 'sub',
                        normalize_features = True,
                        normalize_targets = True,
                        ):
    """
    Extract features and targets from the DataFrames
    
    Input
    ---------
    df_sub: pandas.DataFrame
    n_features: int, default = 7
    time_steps: int, default = 7
    target_attributes: str, default = 'confidence'
    group_col: str, default = 'sub'
    normalize_features: bool, default = True
    normalize_targets: bool, default = True
    
    Output
    ---------
    features: ndarray, (n_samples, n_features)
    targets: ndarray, (nsamples,)
    groups: ndarray, (nsamples,)
    accuracies: ndarray, (nsamples,)
    """
    features            = df_sub[[f"feature{ii + 1}" for ii in range(n_features)]].values
    targets             = df_sub["targets"].values
    if normalize_features:
        # rescale the features between 0 and 1
        features        = features / features.max()
    if normalize_targets:
        targets         = targets / targets.max()
    groups              = df_sub[group_col].values
    accuracies          = df_sub['accuracy'].values
    return features, targets, groups, accuracies

def append_dprime_metadprime(df,df_metadprime):
    temp = []
    for (filename,sub_name),df_sub in tqdm(df.groupby(['filename','sub']),desc='dprime'):
        df_sub
        idx_ = np.logical_and(df_metadprime['sub_names' ] == sub_name,
                              df_metadprime['file'      ] == filename.split('/')[-1],)
        row = df_metadprime[idx_]
        if len(row) > 0:
            df_sub['metadprime'] = row['metadprime'].values[0]
            df_sub['dprime'    ] = row['dprime'    ].values[0]
            temp.append(df_sub)
    df = pd.concat(temp)
    return df

def label_high_low(df,n_jobs = 1):
    """
    to determine the high and low metacognition, the M-ratio should be used
    M-ratio = frac{meta-d'}{d'}
    """
    df['m-ratio'] = df['metadprime'] / (df['dprime']  + 1e-12)
    df_temp = df.groupby(['filename','sub']).mean().reset_index()
    m_ratio = df_temp['metadprime'].values / (df_temp['dprime'].values + 1e-12)
    criterion = np.median(m_ratio)
    df['level']  = df['m-ratio'].apply(lambda x: 'high' if x >= criterion else 'low')

    return df

def resample_ttest(x,
                   baseline         = 0.5,
                   n_ps             = 100,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12, 
                   verbose          = 0,
                   full_size        = True,
                   stat_func        = np.mean,
                   size_catch       = int(1e4),
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Parameters
    ----------
    x : numpy.ndarray, shape (n_samples,)
        the data that is to be compared
    baseline : float, default = 0.5 for ROC AUC
        the single point that we compare the data with
    n_ps : int, default = 100
        number of p values we want to estimate
    n_permutation : int, default = 10000
        number of resampling
    one_tail : bool
        whether to perform one-tailed comparison
    n_jobs : int or None, default = 12
        -1 uses all CPUs
    verbose : int or None, default = 0
    full_size : bool
        exist to control for memory overload when the data is too big
    stat_func : callable, default = numpy.mean
        the function we use to estimate the effect, we could also use median or
        many other statistical estimates
    size_catch : int, default = int(1e4)
        exist to control for memory overload when the data is too big
        
    Return
    ----------------
    ps : float or numpy.ndarray, shape (n_ps,)
    """
    
    import gc
    import numpy as np
    from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = stat_func(x)
    null            = x - stat_func(x) + baseline # shift the mean to the baseline but keep the distribution
    
    if null.shape[0] > size_catch: # catch for big data
        full_size   = False
    if not full_size:
        size        = (size_catch,int(n_permutation))
    else:
        size        = (null.shape[0],int(n_permutation))
    
    gc.collect()
    def t_statistics(null,size,):
        """
        Parameters
        ------
        null : numpy.ndarray,
            shifted data distribution
        size: tuple of 2 integers (n_for_averaging,n_permutation)
        
        Return
        ------
        float \in (0,1]
        """
        null_dist   = np.random.choice(null,size = size,replace = True)
        t_null      = stat_func(null_dist,0)
        if one_tail:
            return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
        else:
            return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) / 2
    if n_ps == 1:
        ps = t_statistics(null, size,)
    else:
        ps = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                        'null':null,
                        'size':size,}) for i in range(n_ps))
        ps = np.array(ps)
    return ps

def resample_ttest_2sample(a,b,
                           n_ps                 = 100,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = 6,
                           verbose              = 0,
                           stat_func            = np.mean,
                           ):
    """
    Parameters
    ---------
    a : ndarray, shape (n_samples,)
    b : ndarray, shape (n_samples,)
    n_ps : int, default = 100
        number of p values to estimate
    n_permutation : in, default = 10000
        numer of resample to estimate one p value
    one_tail : bool, default = False
    match_sample_size : bool, default = True
        whether to perform matching sampleing t test
    n_jobs : int or None, default = 6
    verbose : int or bool, default = 0
    stat_func : callable
    
    Return
    -----------
    ps : float or ndarray, shape (n_ps,)
    """
    from joblib import Parallel,delayed
    import gc
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_ps           = n_ps,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,
                                     stat_func      = stat_func)
        return ps
    else: # when the samples are independent
        t_experiment        = stat_func(a) - stat_func(b)
        if not one_tail:
            t_experiment    = np.abs(t_experiment)
            
        def t_statistics(a,b):
            """
            shuffle the data for both groups
            mix -> shuffle -> split -> compare
            """
            group           = np.concatenate([a,b])
            np.random.shuffle(group)
            new_a           = group[:a.shape[0]]
            new_b           = group[a.shape[0]:]
            t_null          = stat_func(new_a) - stat_func(new_b)
            if not one_tail:
                t_null      = np.abs(t_null)
            return t_null
        
        gc.collect()
        ps = np.zeros(n_ps)
        for ii in range(n_ps):
            t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
            if one_tail:
                ps[ii] = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
            else:
                ps[ii] = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
        return ps

class MCPConverter(object):
    import statsmodels as sms
    """
    https://gist.github.com/naturale0/3915e2def589553e91dce99e69d138cc
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores = None):
        self.pvals                    = pvals
        self.zscores                  = zscores
        self.len                      = len(pvals)
        if zscores is not None:
            srted                     = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals         = srted[:, 0]
            self.sorted_zscores       = srted[:, 1]
        else:
            self.sorted_pvals         = np.array(sorted(pvals.copy()))
        self.order                    = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method           = "holm"):
        import statsmodels as sms
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method == "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method == "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method == "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method == "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods = ["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method != "lfdr":
                    df[method] = self.adjust(method)
        return df

def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'

def get_array_from_dataframe(df,column_name):
    return np.array([item for item in df[column_name].values[0].replace('[',
                     '').replace(']',
                        '').replace('\n',
                          '').replace('  ',
                            ' ').split(' ') if len(item) > 0],
                    dtype = 'float32')

def set_line_lims(dict_condition,ylims = [(-0.325,0.325),(-0.675,0.675)]):
    lims = {list(dict_condition.values())[1]:dict(xticks = np.arange(-3,0),
                     xticklabels = np.arange(-7,-4),
                     ylim = ylims[0]),
        list(dict_condition.values())[0]:dict(xticks = np.arange(-3,0),
                      xticklabels = np.arange(-3,0),
                      ylim = ylims[1])}
    return lims

def get_groupby_average():
    groupby_average = {'confidence':{'LOO':['study_name','decoder','accuracy_train','accuracy_test'],
                                     'cross_domain':['fold','decoder','source','accuracy_train','accuracy_test']},
                       'adequacy':  {'LOO':['study_name','decoder'],
                                     'cross_domain':['filename','decoder','source']}
                       }
    return groupby_average

# def load_results(data_type      = 'confidence', # confidence or adequacy
#                  within_cross   = 'LOO', # LOO or cross_domain
#                  working_data   = [],
#                  dict_rename    = {0:'incorrect trials',1:'correct trials'},
#                  dict_condition = {'past':'T-7,T-6,T-5','recent':'T-3,T-2,T-1'}):
#     # measure: confidence, within perceptual domain decoding
#     if (data_type == 'confidence') and (within_cross == 'LOO'):
#         df                      = []
#         for f in working_data:
#             temp                    = pd.read_csv(f)
#             study_name              = re.findall('\(([^)]+)',f)[0]
#             temp['study_name']      = study_name
#             temp['decoder']         = f.split('/')[-1].split(' ')[0]
#             condition               = f.split(' ')[1]
#             if dict_condition is not None:
#                 temp['condition']   = dict_condition[condition]
#             df.append(temp)
#         df                      = pd.concat(df)
        
#         for col_name in ['accuracy_train','accuracy_test']:
#             df[col_name]        = df[col_name].map(dict_rename)
#         groupby                 = get_groupby_average()[data_type][within_cross]
        
#         if dict_condition is not None:
#             groupby.append('condition')
        
#         # averge within each study
#         df_ave                  = df.groupby(groupby).mean().reset_index()
#         return df_ave
#     # measure: adequacy, within perceptual domain decoding
#     elif (data_type == 'adequacy') and (within_cross == 'LOO'):
#         df                      = []
#         for f in working_data:
#             temp                    = pd.read_csv(f)
#             study_name              = re.findall('\(([^)]+)',f)[0]
#             temp['study_name']      = study_name
#             temp['decoder']         = f.split('/')[-1].split(' ')[0]
#             condition               = f.split(' ')[1]
#             if dict_condition is not None:
#                 temp['condition']   = dict_condition[condition]
#             df.append(temp)
#         df                      = pd.concat(df)
        
#         groupby                 = get_groupby_average()[data_type][within_cross]
#         if dict_condition is not None:
#             groupby.append('condition')
#         # averge within each study
#         df_ave                  = df.groupby(groupby).mean().reset_index()
#         return df_ave
#     # measure: confidence, cross domain decoding
#     elif (data_type == 'confidence') and (within_cross == 'cross_domain'):
#         df                      = []
#         for f in working_data:
#             temp                    = pd.read_csv(f)
#             temp['decoder']         = f.split('/')[-1].split('_')[0]
#             condition               = f.split('_')[-1].split(' ')[0]
#             if dict_condition is not None:
#                 temp['condition']   = dict_condition[condition]
#             df.append(temp)
#         df                          = pd.concat(df)
        
#         for col_name in ['accuracy_train','accuracy_test']:
#             df[col_name]        = df[col_name].map(dict_rename)
        
#         groupby                 = get_groupby_average()[data_type][within_cross]
#         if dict_condition is not None:
#             groupby.append('condition')
#         # averge within each study
#         df_ave                  = df.groupby(groupby).mean().reset_index()
#         return df_ave
#     # measure: adequacy, cross domain decoding
#     elif (data_type == 'adequacy') and (within_cross == 'cross_domain'):
#         df                      = []
#         for f in working_data:
#             temp                    = pd.read_csv(f)
#             temp['decoder']         = f.split('/')[-1].split('_')[0]
#             condition               = f.split('_')[-1].split(' ')[0]
#             if dict_condition is not None:
#                 temp['condition']   = dict_condition[condition]
#             df.append(temp)
#         df                          = pd.concat(df)
        
#         groupby                     = get_groupby_average()[data_type][within_cross]
#         if dict_condition is not None:
#             groupby.append('condition')
#         # averge within each study
#         df_ave                      = df.groupby(groupby).mean().reset_index()
#         return df_ave