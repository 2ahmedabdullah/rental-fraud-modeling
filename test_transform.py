#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:46:50 2021

"""

import numpy as np
import pandas as pd
from numpy import nan
from glob import glob
import yaml
from utils import *
from sklearn.pipeline import Pipeline

#exit()
#import pdb;pdb.set_trace()


data_path = './data/'

def validation_transform(val):
    y = val['y']

    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
        print(variables)

    n = variables['NUMERIC_VARS']
    mappings = variables['MAPPINGS']
    rare_categories = variables['RARE_CATEGORIES']
    numeric_vars = variables['NUMERIC_VARS']
    new_cat_vars = variables['CATEGORIC_VARS']

    numeric_val= val[numeric_vars]
    categoric_val= val[new_cat_vars]
    
    numeric_val_filled = transform_numeric(numeric_val)
    
    categoric_filled = transform_categoric(categoric_val,rare_categories, mappings)
    
    categoric_filled = categoric_filled.astype(int)
    
    val_x = pd.concat([numeric_val_filled, categoric_filled], axis=1)
    
    val_y= pd.Series(np.where(y.values == 'good', 1, 0),y.index)

    val_x.to_csv(data_path+'xtest.csv', index=False)
    val_y.to_csv(data_path+'ytest.csv', index=False)
    print('success')
    

if __name__ == '__main__':
    #enter the TEST FILE PATH HERE-------->
    val = pd.read_csv(data_path+'validation.csv', sep = ';')
    t = validation_transform(val)

