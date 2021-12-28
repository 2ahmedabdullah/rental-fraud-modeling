#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:43:38 2021

"""

from __future__ import division, print_function #python2.0

__version__ = '1.0'
__author__ = 'pL'

import pandas as pd
from numpy import nan
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import yaml
from utils import *


#exit()
#import pdb;pdb.set_trace()


data_path = './data/'
plot_path = './plots/'


if __name__ == '__main__':

    #context manager LOADING VARIABLES FROM FROM YAML FILE
    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
        print(variables)
    n = variables['N']
    mappings = variables['MAPPINGS']
    corr_coeff = variables['CORR_COEFF']
    thresh = variables['THRESH']

    #LOAD THE TRAINING DATA
    
    file = glob(data_path+'training.csv')[0]
    train = pd.read_csv(file, sep = ';')
    y = train['y']


    #TARGET CLASS COUNTS 
    y.value_counts()
    train['y'].value_counts().plot.bar(figsize=(3, 3), rot=0)
    plt.suptitle('Class wise distribution', fontsize=18)
    plt.xlabel('class', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'class_wise_distribution.png')
    plt.show()


    #DROPPING TARGET COLUMN AND ID
    train = train.drop('y', axis=1)
    train = train.drop('Unnamed: 0', axis=1)

    #PLOTTING CARDINALITY OF EACH FEATURE
    unique_values = train.nunique().sort_values(ascending=False)

    unique_values.plot.bar(figsize=(9, 3), rot=0)
    plt.axhline(y=n, color='r', linestyle='-')
    plt.suptitle('Cardinality of Each Feature', fontsize=18)
    plt.xlabel('features', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'cardinality.png')    
    plt.show()


    #SPLITTING NUMERIC AND CATERGORIC FEATURES

    categorical_var = [var for var in train.columns if train[var].nunique() <=n]
    numerical_var = [var for var in train.columns if train[var].nunique() >n]
    categoric_data = train[categorical_var]
    numeric_data = train[numerical_var]

    # NUMERIC DATAFRAME TRANSFORMATION


    numeric_data = numeric_transform1(numeric_data)

    # VISUALIZING NAN IN EACH FEATURE
    nan_mean = numeric_data.isna().mean().sort_values(ascending=False)
    nan_mean.plot.bar(figsize=(6, 3), rot=0)
    plt.suptitle('Percentage of NaN in Numeric Features', fontsize=18)
    plt.xlabel('features', fontsize=10)
    plt.ylabel('percent', fontsize=10)
    plt.savefig(plot_path+'NaN_in_numeric_feature.png')    
    plt.show()


    #correlation
    corr_matrix = numeric_data.corr()
    corr_matrix.style.background_gradient(cmap='coolwarm')
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_feature2 = [column for column in upper.columns if any(upper[column] > corr_coeff)]


    #WE DROP HIGHLY CORRELATED FEATURES WHERE corr>0.9
    numeric_data.drop(drop_feature2, axis=1, inplace=True)
    final_selected_numeric_features=list(numeric_data)

    #GROUPBY MEANS OF EACH NUMERIC FEATURE WRT TO TARGET CLASS
    for j in range(0,len(final_selected_numeric_features)):
        numeric_data.groupby(y, as_index=False)[final_selected_numeric_features[j]].mean().plot(kind='bar')

    #filling NAs with group means <CLASS MEAN NA FILLING>
    numeric_filled = numeric_data.groupby(y).transform(lambda x: x.fillna(x.mean()))

    #================CATEGORICAL FEATURES DATA VISULAIZATION===========================================

    #VISAULAZING EACH CATEGORICAL FEATURE DISTRIBUTION
    for j in range(0,len(categorical_var)):
        categoric_data[categorical_var[j]].value_counts().plot(kind='bar')
        plt.title(categorical_var[j])
        plt.savefig(plot_path+'cat_'+categorical_var[j]+'_distribution.png')
        plt.show()


    #WE SEE HOW CLASS WISE DISTRIBUTION OF X.5 AND X.10
    categoric_data.groupby(y)['x.5'].value_counts().plot.bar(figsize=(9, 3))
    plt.axvline(x=14.5, color='r', linestyle='-')
    plt.suptitle('x.5 class wise distribution', fontsize=18)
    plt.xlabel('categories', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'x.5_distribution.png')
    plt.show()

    categoric_data.groupby(y)['x.10'].value_counts().plot.bar(figsize=(9, 3))
    plt.axvline(x=10.5, color='r', linestyle='-')
    plt.suptitle('x.10 class wise distribution', fontsize=18)
    plt.xlabel('categories', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'x.10_distribution.png')
    plt.show()


    categoric_data['x.6'].value_counts().plot.bar(figsize=(6, 3), rot=0)
    plt.axhline(y=thresh*len(train), color='r', linestyle='-')
    plt.suptitle('x.6 class wise distribution', fontsize=18)
    plt.xlabel('categories', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'x.6_distribution.png')
    plt.show()



    #filling and SUMMARIZING RARE categories
    transformed_categoric_data, rare_categories = replace_rare_categories(categoric_data)

    #VISULAIZATION OF CLASS WISE DISTRIBUTION AFTER REPLACING WITH RARE VALUES
    transformed_categoric_data.groupby(y)['x.10'].value_counts().plot.bar(figsize=(12, 6))
    plt.axvline(x=5.5, color='r', linestyle='-')
    plt.suptitle('x.10 class wise distribution', fontsize=18)
    plt.xlabel('categories', fontsize=15)
    plt.ylabel('counts', fontsize=15)
    plt.savefig(plot_path+'x.10_summarized_distribution.png')
    plt.show()


    transformed_categoric_data['x.6'].value_counts().plot.bar(figsize=(6, 3), rot=0)
    plt.axhline(y=thresh*len(train), color='r', linestyle='-')
    plt.suptitle('x.6 class wise distribution', fontsize=18)
    plt.xlabel('categories', fontsize=10)
    plt.ylabel('counts', fontsize=10)
    plt.savefig(plot_path+'x.6_summarized_distribution.png')
    plt.show()

    #WE APPLY MAPPINGS TO THE CATEGORICAL FEATURES
    categoric_filled = categoric_transform1(transformed_categoric_data,mappings)
    categoric_filled = categoric_filled.astype(int)


    #SAVING TRANSFORMED TRIAN DATA
    train_x = pd.concat([numeric_filled, categoric_filled], axis=1)
    train_y= pd.Series(np.where(y.values == 'good', 1, 0),y.index)


    train_x.to_csv(data_path+'xtrain.csv', index=False)
    train_y.to_csv(data_path+'ytrain.csv', index=False)




