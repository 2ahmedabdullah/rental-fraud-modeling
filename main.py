#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 20:21:01 2021

"""

import pandas as pd
import numpy as np
import yaml
from utils import *

data_path = './data/'
model_path = './saved_models/'
plot_path = './plots/'

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #we load the transformed data from previous steps
    x_train = pd.read_csv(data_path+'xtrain.csv')
    y_train = pd.read_csv(data_path+'ytrain.csv')
    x_test = pd.read_csv(data_path+'xtest.csv')
    y_test = pd.read_csv(data_path+'ytest.csv')
  
    #UNCOMMENT AND RUN THE MODEL IF YOU WANT TO TRIAN THE NEURAL NETWORK   
    #nn = neural_network(x_train,y_train,x_test,y_test)        
    
    
    xgb = xgboost_model(x_train,y_train,x_test,y_test)
    rf = random_forest(x_train,y_train,x_test,y_test)
    log_reg = logistic(x_train,y_train,x_test,y_test)
    
    #DO PREDICTIONS ON ALREADY TRAINED MODEL
    best1 = best_model(model_path+'new_model1.h5', x_test, y_test,1)
    best2 = best_model(model_path+'new_model2.h5', x_test, y_test,2)
    
    #PLOTTING ROC CURVE    
    plt.plot(xgb[0],xgb[1], label="XGB, AUC="+str(xgb[2]))
    plt.plot(rf[0], rf[1], label="RF, AUC="+str(rf[2]) )
    plt.plot(log_reg[0],log_reg[1], label="Log_Reg, AUC="+str(log_reg[2]))
    plt.plot(best1[0], best1[1], label="Neural_Network1, AUC="+str(best1[2]))
    plt.plot(best2[0], best2[1], label="Neural_Network2, AUC="+str(best2[2]))
    plt.legend(loc=0)
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    plt.savefig(plot_path+'ROC.png')
    plt.show()

