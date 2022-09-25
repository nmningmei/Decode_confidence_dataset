#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:13:31 2022

@author: ning mei
"""
import os,gc,re

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from glob import glob
from matplotlib import pyplot as plt

sns.set_context('paper',font_scale = 2.5)
sns.set_style('white')

orders = ["incorrect trials","correct trials"]
target_order = ['Cognitive','Memory','Mixed']
palette = ['deepskyblue','tomato']
x_order = ['confidence','accuracy',
           'confidence-accuracy','RT',
           'confidence-RT','all']
model_names = ['Support Vector Machine',
               'Random Forest',
              #'Recurrent Neural Network',
               ]
CD_order = ['Perception','Cognitive','Memory','Mixed']
ylabel_dict = {'regression':'Variance explained',
               'classification':'ROC AUC'}
hline_dict = {'regression':0,
               'classification':0.5}
ylim_dict = {'regression':(-.85,.5),
             'classification':(0,1.)}
y_annotation = dict(regression = .45,
                    classification = .875)

def get_model_type(x:str):
    """
    

    Parameters
    ----------
    x : str
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    if 'LinearSV' in x:
        return 'Support Vector Machine'
    elif 'randomforest' in x:
        return 'Random Forest'
    else:
        return 'Recurrent Neural Network'

def load_review_scores(working_dir:str,):
    """
    

    Parameters
    ----------
    working_dir : str
        DESCRIPTION.

    Returns
    -------
    df_scores : TYPE
        DESCRIPTION.

    """
    dfs = {}
    for folder_name in x_order:
        working_data = glob(os.path.join(
            working_dir.replace('replace',folder_name),'*.csv'))
        
        # load the data
        temp = []
        for f in working_data:
            df = pd.read_csv(f).dropna()
            df['model_name'] = df['best_params'].apply(get_model_type)
            col_names = [item for item in df.columns if ('features' in item)]
            if '_RNN_' in f:
                for col_name in col_names:
                    df[col_name] = df[col_name].apply(lambda x: -x)
            temp.append(df)
        dfs[folder_name] = pd.concat(temp)
        
    df_scores = pd.concat([df[['score',
                               'feature_type',
                               'model_name',
                               'source_data',
                               'target_data']] for df in dfs.values()])
    df_scores['score'] = pd.to_numeric(df_scores['score'])
    return df_scores

def load_groupby_data(f:str,average:bool = False,groupby:list = []) -> pd.core.frame.DataFrame:
    """
    

    Parameters
    ----------
    f : str
        DESCRIPTION.
    average : bool, optional
        DESCRIPTION. The default is False.
    groupby : list, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    temp : TYPE
        DESCRIPTION.

    """
    temp = pd.read_csv(f)
    if average and len(groupby) > 0:
        temp = temp.groupby(groupby).mean().reset_index()
    study_name = re.findall(r'\((.*?)\)',f)[0]
    temp['study_name'] = study_name
    if not isinstance(temp,pd.core.frame.DataFrame):
        temp = temp.to_frame().T
    return temp

def load_and_map(working_data:list,
                 average:bool = True,
                 groupby:list = ['accuracy_train','accuracy_test'],
                 ) -> pd.core.frame.DataFrame:
    """
    load the dataframes and then average the concatenated df
    and then map the labels for plotting

    Parameters
    ----------
    working_data : list
        DESCRIPTION.
    average : bool, optional
        DESCRIPTION. The default is True.
    groupby : list, optional
        DESCRIPTION. The default is ['accuracy_train','accuracy_test'].
     : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    # load the data and average by the training and testing sets
    df = pd.concat([load_groupby_data(f,
                                      average = average,
                                      groupby = groupby,
                                      ) for f in working_data])
    # map the correct and incorrect trials
    df['accuracy_train'] = df['accuracy_train'].map({0:'incorrect trials',
                                                     1:'correct trials'})
    df['accuracy_test'] = df['accuracy_test'].map({0:'incorrect trials',
                                                   1:'correct trials'})
    return df

def figure1(working_data:list,
            model_name:str,
            average:bool = True,
            groupby:list = ['accuracy_train',
                            'accuracy_test'],
            figure_name = 'figure1',
            ):
    """
    load the data and then plot the results

    Parameters
    ----------
    working_data : list
        DESCRIPTION.
    model_name : str
        DESCRIPTION.
    average : bool, optional
        DESCRIPTION. The default is True.
    groupby : list, optional
        DESCRIPTION. The default is ['accuracy_train','accuracy_test'].
    figure_name : TYPE, optional
        DESCRIPTION. The default is 'figure1'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    p : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    """
    df = load_and_map(working_data,
                      average = average,
                      groupby = groupby,
                      )
    # plot
    # fig,ax = plt.subplots(figsize = (8,6))
    # p = so.Plot(df,
    #             x = 'accuracy_train',
    #             y = 'score',
    #             fill = 'accuracy_test',
    #             ).add(so.Bar(alpha = .5), 
    #                   so.Agg(),
    #                   so.Dodge(),
    #                   ).add(so.Range(),
    #                         so.Est(errorbar = 'se',seed = 12345),
    #                         so.Dodge()
    #                         ).label(x = 'Training data',
    #                                 y = 'ROC AUC',
    #                                 fill = 'Testing data').layout(
    #                                     engine="constrained",
    #                                     ).on(ax).plot()
    # ax.set(ylim = (0.4,0.8))
    # ax.axhline(0.5, linestyle = '--', color = 'black',alpha = .7)
    # for x in [0 - .2, 0 + .2, 1 - .2, 1 + .2]:
    #     ax.annotate('***',
    #             xy = (x,.7),
    #             ha = 'center',
    #             fontsize = 14)
    fig,ax = plt.subplots(figsize = (10,8))
    ax = sns.barplot(x = 'accuracy_train',
                     y = 'score',
                     data = df,
                     hue = 'accuracy_test',
                     errorbar = 'se',#('ci',95),
                     seed = 12345,
                     palette = palette,
                     )
    ax.legend(loc = 'upper right')
    ax.get_legend().set_title("Testing data")
    ax.set(ylim = (0.4,0.85),
           xlabel = 'Training data',
           ylabel = 'ROC AUC',
           )
    ax.axhline(0.5, linestyle = '--', color = 'black',alpha = .7)
    for x in [0 - .2, 0 + .2, 1 - .2, 1 + .2]:
        ax.annotate('***',
                xy = (x,.7),
                ha = 'center',
                fontsize = 14)
    fig.savefig(os.path.join(figure_dir, 
                             f'{figure_name}.{model_name} decoding results.jpg'),
            dpi = 300,
            bbox_inches = 'tight')
    return fig,ax,df

def figure2(working_data_past:list,
            working_data_recent:list,
            model_name:str,
            average:bool = True,
            groupby:list = ['accuracy_train',
                            'accuracy_test'],
            figure_name = 'figure2',):
    """
    

    Parameters
    ----------
    working_data_past : list
        DESCRIPTION.
    working_data_recent : list
        DESCRIPTION.
    model_name : str
        DESCRIPTION.
    average : bool, optional
        DESCRIPTION. The default is True.
    groupby : list, optional
        DESCRIPTION. The default is ['accuracy_train','accuracy_test'].
    figure_name : TYPE, optional
        DESCRIPTION. The default is 'figure2'.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    df_plot : TYPE
        DESCRIPTION.

    """
    # load the data and average by the training and testing sets
    df_past = load_and_map(working_data_past,
                           average = average,
                           groupby = groupby,
                           )
    df_past['split'] = "T-7, T-6, T-5"
    df_recent = load_and_map(working_data_recent,
                             average = average,
                             groupby = groupby,
                             )
    df_recent['split'] = "T-3, T-2, T-1"
    
    df_plot = pd.concat([df_past,df_recent])
    
    g = sns.catplot(x = 'accuracy_train',
                    y = 'score',
                    data = df_plot,
                    col = 'accuracy_test',
                    col_order = ["incorrect trials","correct trials"],
                    hue = 'split',
                    hue_order = ["T-7, T-6, T-5","T-3, T-2, T-1"],
                    kind = 'bar',
                    seed = 12345,
                    palette = palette,
                    errorbar = 'se',
                    )
    g.set_axis_labels('Training data','ROC AUC').set(
        ylim = (.4,.85))
    g._legend.set_title('')
    [ax.axhline(0.5,linestyle = '--',color = 'black',alpha = .7) for ax in g.axes.flatten()]
    g.axes.flatten()[0].set(title = 'Tested on incorrect trials')
    g.axes.flatten()[1].set(title = 'Tested on correct trials')
    for ax in g.axes.flatten():
        for x in [0 - .2, 0 + .2, 1 - .2, 1 + .2]:
            ax.annotate('**',
                    xy = (x,.72),
                    ha = 'center',
                    fontsize = 14,
                    )
        for x_center in [0,1]:
            ax.plot([x_center - .2,
                     x_center - .2,
                     x_center + .2,
                     x_center + .2,],
                    [.75,.77,.77,.75],
                    color = 'black',
                    )
            ax.annotate('ooo',
                        xy = (x_center,.78),
                        ha = 'center',
                        fontsize = 14,
                        )
    g.savefig(os.path.join(figure_dir,
                           f'{figure_name}.{model_name} decoding results.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    return fig,ax,df_plot

def figure4(working_data:list,
            model_name:str = 'RF',
            figure_name:str = 'figure4',
            condition:str = 'cross_domain',
            ):
    """
    

    Parameters
    ----------
    working_data : list
        DESCRIPTION.
    model_name : str, optional
        DESCRIPTION. The default is 'RF'.
    figure_name : str, optional
        DESCRIPTION. The default is 'figure4'.
    condition : str, optional
        DESCRIPTION. The default is 'cross_domain'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    g : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.
    df_stats : TYPE
        DESCRIPTION.

    """
    df = pd.concat([pd.read_csv(f) for f in working_data])
    df['accuracy_train'] = df['accuracy_train'].map({0:'incorrect trials',
                                                     1:'correct trials'})
    df['accuracy_test'] = df['accuracy_test'].map({0:'incorrect trials',
                                                   1:'correct trials'})
    df['source'] = df['source'].map(dict(cognitive='Cognitive',
                                         mem_4='Memory',
                                         mixed_4='Mixed'))
    working_dir = '../../decoding_confidence_dataset/stats/confidence'
    df_stats = pd.read_csv(os.path.join(working_dir,
                                  condition,
                                  'scores.csv'))
    df_stats = df_stats[df_stats['decoder'] == model_name]
    df_stats = df_stats.sort_values(['accuracy_test','accuracy_train'],
                                    ascending = False,)
    df_stats = pd.concat([df_stats[df_stats['target_data'] == target_data] for target_data in target_order])
    
    g = sns.catplot(data = df,
                    x = 'accuracy_train',
                    y = 'score',
                    hue = 'accuracy_test',
                    hue_order = orders,
                    order = orders,
                    col = 'source',
                    col_order = target_order,
                    palette = palette,
                    seed = 12345,
                    kind = 'bar',
                    aspect = 1.5,
                    errorbar = 'se',
                    )
    g.set_axis_labels('Training data','ROC AUC').set_titles('{col_name}')
    g.set(ylim = (0.45,.7))
    g._legend.set_title("Testing data")
    [ax.axhline(0.5,linestyle = '--',color = 'black',alpha = .7) for ax in g.axes.flatten()]
    for (target_data,df_sub),ax in zip(df_stats.groupby('target_data'),g.axes.flatten()):
        for x,(ii,row) in zip([0 - .2, 0 + .2, 1 - .2, 1 + .2],
                          df_sub.iterrows()):
            ax.annotate(row['stars'],
                    xy = (x,.65),
                    ha = 'center',
                    fontsize = 14)
    g.savefig(os.path.join(figure_dir,
                           f'{figure_name}.{model_name} decoding results.jpg'))
    return g,df,df_stats

def figure5(working_dir:str,
            condition:str = 'cross_domain',
            model_name:str = 'RF',
            figure_name:str = 'figure5',
            ):
    """
    

    Parameters
    ----------
    working_dir : str
        DESCRIPTION.
    condition : str, optional
        DESCRIPTION. The default is 'cross_domain'.
    model_name : str, optional
        DESCRIPTION. The default is 'RF'.
    figure_name : str, optional
        DESCRIPTION. The default is 'figure5'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    g : TYPE
        DESCRIPTION.
    df_plot : TYPE
        DESCRIPTION.
    df_stats : TYPE
        DESCRIPTION.
    df_stats_paired : TYPE
        DESCRIPTION.

    """
    df_past = pd.read_csv(os.path.join(working_dir,
                                       condition,
                                       f'{model_name}_confidence_past results.csv'))
    df_past['split'] = "T-7, T-6, T-5"
    df_recent = pd.read_csv(os.path.join(working_dir,
                                         condition,
                                         f'{model_name}_confidence_recent results.csv'))
    df_recent['split'] = "T-3, T-2, T-1"
    df_plot = pd.concat([df_past,df_recent])
    
    df_plot['accuracy_train'] = df_plot['accuracy_train'].map({0:'incorrect trials',
                                                     1:'correct trials'})
    df_plot['accuracy_test'] = df_plot['accuracy_test'].map({0:'incorrect trials',
                                                   1:'correct trials'})
    df_plot['source'] = df_plot['source'].map(dict(cognitive='Cognitive',
                                                   mem_4='Memory',
                                                   mixed_4='Mixed'))
    working_dir = '../../decoding_confidence_dataset/stats/confidence'
    df_stats = pd.read_csv(os.path.join(working_dir,
                                       condition,
                                       'scores_split.csv'))
    df_stats = df_stats[df_stats['decoder'] == model_name]
    df_stats = df_stats.sort_values(['accuracy_test',
                                     'accuracy_train',
                                     'condition'],
                                    ascending = False,)
    df_stats = pd.concat([df_stats[df_stats['target_data'] == target_data] for target_data in target_order])
    df_stats_paired = pd.read_csv(os.path.join(working_dir,
                                               condition,
                                               'scores_paired.csv'))
    df_stats_paired = df_stats_paired[df_stats_paired['decoder'] == model_name]
    df_stats_paired = df_stats_paired.sort_values(['accuracy_test',
                                                   'accuracy_train',],
                                                  ascending = False,)
    df_stats_paired = pd.concat([df_stats_paired[df_stats_paired['source'] == target_data] for target_data in target_order])
    
    g = sns.catplot(x = 'accuracy_train',
                    y = 'score',
                    data = df_plot,
                    col = 'accuracy_test',
                    col_order = ["incorrect trials","correct trials"],
                    row = 'source',
                    row_order = target_order,
                    hue = 'split',
                    hue_order = ["T-7, T-6, T-5","T-3, T-2, T-1"],
                    kind = 'bar',
                    seed = 12345,
                    palette = palette,
                    aspect = 2,
                    errorbar = 'se',
                    )
    g.set_axis_labels('Training data','ROC AUC').set_titles(
        '{row_name} | Tested on {col_name}').set(
        ylim = (.4,.85))
    [ax.axhline(0.5,linestyle = '--',color = 'black',alpha = .7) for ax in g.axes.flatten()]
    g._legend.set_title("")
    
    for row_axes,target_data in zip(g.axes,target_order):
        for col_axes,accuracy_test in zip(row_axes,["incorrect trials","correct trials"]):
            df_stats_sub = df_stats[np.logical_and(df_stats['target_data'] == target_data,
                                                   df_stats['accuracy_test'] == accuracy_test)]
            df_paired_sub = df_stats_paired[np.logical_and(df_stats_paired['source'] == target_data,
                                                           df_stats_paired['accuracy_test'] == accuracy_test)]
            for x,(ii,row) in zip([0 - .2, 0 + .2, 1 - .2, 1 + .2],
                             df_stats_sub.iterrows()):
                col_axes.annotate(row['stars'],
                                  xy = (x,.68),
                                  ha = 'center',
                                  fontsize = 14)
            for ii,row in df_paired_sub.reset_index().iterrows():
                sign = row['stars'].replace('*','o') if '*' in row['stars'] else row['stars']
                col_axes.plot([ii - .2,
                               ii - .2,
                               ii + .2,
                               ii + .2,],
                              [.72,.75,.75,.72],
                              color = 'black',
                              )
                col_axes.annotate(sign,
                                  xy = (ii,.78),
                                  ha = 'center',
                                  fontsize = 14,
                                  )
    g.savefig(os.path.join(figure_dir,
                           f'{figure_name} {model_name}.jpg'),
              dpi = 300,
              bbox_inches = 'tight')
    return g,df_plot,df_stats,df_stats_paired

def reviewer_plot(working_dir:str,
                  major_type:str = 'regression',
                  figure_name:str = 'supfigure6.1',):
    df_score = load_review_scores(working_dir)
    df_stat = pd.read_csv(f'../stats/scores_{major_type}.csv')
    
    g = sns.catplot(x           = 'feature_type',
                    y           = 'score',
                    order       = x_order,
                    hue         = 'model_name',
                    hue_order   = model_names,
                    row         = 'target_data',
                    row_order   = CD_order,
                    col         = 'source_data',
                    col_order   = CD_order,
                    data        = df_score,
                    kind        = 'bar',
                    aspect      = 2,
                    seed        = 12345,
                    palette     = palette,
                    errorbar    = 'se',
                    )
    xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
    (g.set(xlabel = '',
           ylabel = ylabel_dict[major_type],
           ylim = ylim_dict[major_type])
      .set_titles("{col_name} --> {row_name}"))
    [ax.axhline(hline_dict[major_type],linestyle = '--',color = 'black',alpha = 0.7) for ax in g.axes.flatten()]
    [ax.set_xticklabels(x_order,rotation = 45) for ax in g.axes[-1]]
    for row_axes,target_data in zip(g.axes,CD_order):
        for col_axes,source_data in zip(row_axes,CD_order):
            print(col_axes.title,source_data,target_data)
            df_stat_sub = df_stat[np.logical_and(df_stat['target_data'] == target_data,
                                                 df_stat['source_data'] == source_data)
                                                 ]
            for xtick_obj in xtick_order:
                position        = xtick_obj.get_position()
                xtick_label     = xtick_obj.get_text()
                df_sub_sub = df_stat_sub[df_stat_sub['feature_type'] == xtick_label].sort_values(['model_name'],ascending = False)
                for x_adjustment,(ii,row) in zip([-.2,.2],
                                 df_sub_sub.reset_index().iterrows()):
                    if '*' in row['stars']:
                        col_axes.annotate(row['stars'],
                                          xy = (x_adjustment + position[0],y_annotation[major_type]),
                                          ha = 'center',
                                          fontsize = 14)
    g.savefig(f'../figures/final_figures/{figure_name} {major_type}.jpg',
              dpi = 300,
              bbox_inches = 'tight')
    return g,df_score,df_stat

if __name__ == "__main__":
    figure_dir = '../figures/final_figures'
    
    """
    plot the RF model results as the main results
    """
    """
    1. Decoding confidence levels within the perceptual domain
    """
    # old result folder, from another github repo
    condition = 'LOO'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    # for different models
    model_name = 'RF'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name} cross*csv'))
    fig,ax,df_rf = figure1(working_data,model_name,figure_name = 'figure1')
    model_name = 'SVM'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name} cross*csv'))
    fig,ax,df_svm = figure1(working_data,model_name,figure_name = 'supfigure1.1')
    model_name = 'RNN'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name} cross*csv'))
    fig,ax,df_svm = figure1(working_data,model_name,figure_name = 'supfigure1.2')

    """
    2. Decoding of confidence within the perceptual domain based on recent vs. past trials back.
    """
    # old result folder, from another github repo
    condition = 'LOO'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    # for different models
    model_name = 'RF'
    working_data_past = glob(os.path.join(working_dir,condition,
                                          f'{model_name} past*csv'))
    working_data_recent = glob(os.path.join(working_dir,condition,
                                            f'{model_name} recent*csv'))
    fig,ax,df_rf = figure2(working_data_past,
                           working_data_recent,
                           model_name = model_name,
                           figure_name = 'figure2',
                           )
    model_name = 'SVM'
    working_data_past = glob(os.path.join(working_dir,condition,
                                          f'{model_name} past*csv'))
    working_data_recent = glob(os.path.join(working_dir,condition,
                                            f'{model_name} recent*csv'))
    fig,ax,df_rf = figure2(working_data_past,
                           working_data_recent,
                           model_name = model_name,
                           figure_name = 'supfigure2.1',
                           )
    model_name = 'RNN'
    working_data_past = glob(os.path.join(working_dir,condition,
                                          f'{model_name} past*csv'))
    working_data_recent = glob(os.path.join(working_dir,condition,
                                            f'{model_name} recent*csv'))
    fig,ax,df_rf = figure2(working_data_past,
                           working_data_recent,
                           model_name = model_name,
                           figure_name = 'supfigure2.2',
                           )
    """
    3. Illustration of the feature importance estimates of the RF classifiers within the perceptual domain
    """
    # old result folder, from another github repo
    condition = 'LOO'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    model_name = 'RF'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name} cross*csv'))
    df = load_and_map(working_data,)
    working_dir = '../../decoding_confidence_dataset/stats/confidence'
    df_stats = pd.read_csv(os.path.join(working_dir,
                                  condition,
                                  'feature_importance_secondary.csv'))
    df_plot = pd.melt(df,id_vars = ['accuracy_test',
                                    'accuracy_train',
                                    'sub_name',
                                    'n_sample',
                                    'study_name'],
                      value_vars = [f'feature importance T-{ii}' for ii in np.arange(7,0,-1)],
                      )
    df_plot['variable'] = df_plot['variable'].apply(lambda x:x[-3:])
    g = sns.catplot(x = 'variable',
                    y = 'value',
                    order = [f'T-{ii}' for ii in np.arange(7,0,-1)],
                    hue = 'accuracy_test',
                    hue_order = orders,
                    col = 'accuracy_train',
                    col_order = orders,
                    data = df_plot,
                    seed = 12345,
                    kind = 'bar',
                    aspect = 2,
                    palette = palette,
                    errorbar = 'se',
                    )
    xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
    for acc_train,ax in zip(orders,g.axes.flatten()):
        df_sub = df_stats[df_stats['accuracy_train'] == acc_train]
        df_sub['Time'] = df_sub['Time'].apply(lambda x:x[-3:])
        for ii,text_obj in enumerate(xtick_order):
            position        = text_obj.get_position()
            xtick_label     = text_obj.get_text()
            df_sub_stats    = df_sub[df_sub['Time'] == xtick_label].sort_values(['accuracy_test'],ascending = False)
            for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.2,0.2]):
                print(temp_row['star'])
                ax.annotate(temp_row['star'],
                            xy          = (ii + adjustment,.09),
                            ha          = 'center',
                            fontsize    = 14)
        for ii,(acc_train,df_plot_sub) in enumerate(
                                        df_plot[df_plot['accuracy_train'] == acc_train].groupby(['accuracy_test'])
                                                    ):
            df_plot_sub['x'] = df_plot_sub['variable'].apply(lambda x:7 - int(x[-1]))
            ax = sns.regplot(data = df_plot_sub,
                              x = 'x',
                              y = 'value',
                              seed = 12345,
                              scatter = False,
                              color = palette[ii],
                              ax = ax,
                              )
    g.set_axis_labels('Trial','Feature importance',)
    g.axes.flatten()[0].set(title = 'Trained on incorrect trials')
    g.axes.flatten()[1].set(title = 'Trained on correct trials')
    g.set(ylim = (-0.01,0.1),)
    g._legend.set_title("Testing trials")
    g.savefig(os.path.join(figure_dir,
                             'figure3.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    """
    4. Cross-domain decoding scores of confidence predictions.
    """
    condition = 'cross_domain'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    model_name = 'RF'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name}_confidence fold*csv'))
    g,df_rf,df_stats_rf = figure4(working_data,
                                  model_name = model_name,
                                  )
    model_name = 'SVM'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name}_confidence results.csv'))
    g,df_svm,df_stats_svm = figure4(working_data,
                                    model_name = model_name,
                                    figure_name = 'supfigure4.1',
                                    )
    model_name = 'RNN'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name}_confidence results.csv'))
    g,df_rnn,df_stats_rnn = figure4(working_data,
                                    model_name = model_name,
                                    figure_name = 'supfigure4.2',
                                    )
    """
    5. Decoding confidence across domains based on recent vs. past trials.
    """
    condition = 'cross_domain'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    model_name = 'RF'
    _ = figure5(working_dir,condition,model_name,figure_name = 'figure5')
    model_name = 'SVM'
    _ = figure5(working_dir,condition,model_name,figure_name = 'supfigure5.1')
    model_name = 'RNN'
    _ = figure5(working_dir,condition,model_name,figure_name = 'supfigure5.2')
    """
    6. Illustration of the feature importance estimates of the RF classifiers in the cross-domain decoding.
    """
    # old result folder, from another github repo
    condition = 'cross_domain'
    working_dir = '../../decoding_confidence_dataset/results/confidence'
    model_name = 'RF'
    working_data = glob(os.path.join(working_dir,condition,
                                     f'{model_name}_confidence fold*csv'))
    df = pd.concat([pd.read_csv(f) for f in working_data])
    df['accuracy_train'] = df['accuracy_train'].map({0:'incorrect trials',
                                                     1:'correct trials'})
    df['accuracy_test'] = df['accuracy_test'].map({0:'incorrect trials',
                                                   1:'correct trials'})
    df['source'] = df['source'].map(dict(cognitive='Cognitive',
                                         mem_4='Memory',
                                         mixed_4='Mixed'))
    working_dir = '../../decoding_confidence_dataset/stats/confidence'
    df_stats = pd.read_csv(os.path.join(working_dir,
                                        condition,
                                        'feature_importance_secondary.csv'))
    df_stats['source'] = df_stats['source'].map(dict(cognitive='Cognitive',
                                                     mem_4='Memory',
                                                     mixed_4='Mixed'))
    
    df_plot = pd.melt(df,id_vars = ['accuracy_test',
                                    'accuracy_train',
                                    'sub_name',
                                    'n_sample',
                                    'source',
                                    'filename'],
                      value_vars = [f'feature importance T-{ii}' for ii in np.arange(7,0,-1)],
                      )
    df_plot['variable'] = df_plot['variable'].apply(lambda x:x[-3:])
    
    g = sns.catplot(x = 'variable',
                    y = 'value',
                    order = [f'T-{ii}' for ii in np.arange(7,0,-1)],
                    hue = 'accuracy_test',
                    hue_order = orders,
                    col = 'accuracy_train',
                    col_order = orders,
                    row = 'source',
                    row_order = target_order,
                    data = df_plot,
                    seed = 12345,
                    kind = 'bar',
                    aspect = 2,
                    palette = palette,
                    errorbar = 'se',
                    )
    xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
    for row_axes,target_data in zip(g.axes,target_order):
        for col_axes,accuracy_train in zip(row_axes,["incorrect trials","correct trials"]):
            df_sub = df_stats[np.logical_and(df_stats['accuracy_train'] == accuracy_train,
                                             df_stats['source'] == target_data)]
            df_sub['Time'] = df_sub['Time'].apply(lambda x:x[-3:])
            for ii,text_obj in enumerate(xtick_order):
                position        = text_obj.get_position()
                xtick_label     = text_obj.get_text()
                df_sub_stats    = df_sub[df_sub['Time'] == xtick_label].sort_values(['accuracy_test'],ascending = False)
                for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.2,0.2]):
                    print(temp_row['star'])
                    col_axes.annotate(temp_row['star'],
                                      xy          = (ii + adjustment,.045),
                                      ha          = 'center',
                                      fontsize    = 14)
            for ii,(accuracy_test,df_plot_sub) in enumerate(
            df_plot[df_plot['accuracy_train'] == accuracy_train].groupby(['accuracy_test'])
                                                        ):
                df_plot_sub['x'] = df_plot_sub['variable'].apply(lambda x:7 - int(x[-1]))
                col_axes = sns.regplot(data = df_plot_sub,
                                  x = 'x',
                                  y = 'value',
                                  seed = 12345,
                                  scatter = False,
                                  color = palette[ii],
                                  ax = col_axes,
                                  )
    g.set_axis_labels('Trial','Feature importance',).set_titles(
        '{row_name} | Trained on {col_name}')
    g.set(ylim = (-0.01,0.05),)
    g._legend.set_title("Testing trials")
    g.savefig(os.path.join(figure_dir,'figure6.jpg'),
              dpi = 300,
              bbox_inches = 'tight')
    """
    for reviewer
    """
    major_type = 'regression'
    working_dir = f'../results/{major_type}/replace/*'
    g,df_reg,df_stat_reg = reviewer_plot(working_dir,
                                         major_type,
                                         'supfigure6.1')
    major_type = 'classification'
    working_dir = f'../results/{major_type}/replace/*'
    g,df_clf,df_stat_clf = reviewer_plot(working_dir,
                                         major_type,
                                         'supfigure6.2')




