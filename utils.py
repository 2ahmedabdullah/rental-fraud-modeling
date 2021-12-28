from numpy import nan
import pandas as pd
import yaml
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import roc_auc_score
from sklearn import metrics

model_path = './saved_models/'

def numeric_transform1(numeric_data):
	numeric_data[numeric_data == "?" ] = nan
	numeric_data[numeric_data == "t" ] = nan
	numeric_data[numeric_data == "f" ] = nan

	numeric_data['x.1'] = numeric_data['x.1'].str.replace(',', '.').astype(float)
	numeric_data['x.17'] = numeric_data['x.17'].str.replace(',', '.').astype(float)
	numeric_data['x.18'] = numeric_data['x.18'].str.replace(',', '.').astype(float)
	numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
	return numeric_data


def transform_numeric(numeric_dataframe):
    numeric_dataframe[numeric_dataframe == "?" ] = nan
    numeric_dataframe[numeric_dataframe == "t" ] = nan
    numeric_dataframe[numeric_dataframe == "f" ] = nan
    
    numeric_dataframe['x.1'] = numeric_dataframe['x.1'].str.replace(',', '.').astype(float)
    
    numeric_dataframe = numeric_dataframe.apply(pd.to_numeric, errors='coerce')
    numeric_dataframe = numeric_dataframe.fillna(numeric_dataframe.mean())
    return numeric_dataframe

def replace_rare_categories(categoric_data):
    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
        print(variables)
    percent_thresh = variables['PERCENT_THRESH']


    categorical_var = list(categoric_data)
    transformed_categoric = pd.DataFrame()
    rare = []
    for j in range(0,len(categorical_var)):
        feat = categoric_data[categorical_var[j]]
        b = feat.value_counts()/len(categoric_data)*100
        filt = b<percent_thresh
        rare_cats = list(filt[filt].index)
        val =['rare']*len(rare_cats)
        dictionary = dict(zip(rare_cats, val))
        feat1 = feat.replace(dictionary) 
        transformed_categoric[categorical_var[j]]=feat1
        rare.append(rare_cats)
    return transformed_categoric, rare


def categoric_transform1(data,mappings):
    new_cat_vars=list(data)
    for j in range(0,len(new_cat_vars)):
        var = new_cat_vars[j]
        maps = mappings[var]
        data[var] = data[var].replace(maps) 
    return data


def transform_categoric(data, rare_categories, mappings):
    new_cat_vars=list(data)
    for j in range(0,len(new_cat_vars)):
        var = new_cat_vars[j]
        maps = rare_categories[var]
        data[var] = data[var].replace(maps)
        maps1 = mappings[var]
        data[var] = data[var].replace(maps1) 
    return data

def prediction(model, x_test):
    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
    print(variables)
    binary_threshold = variables['BINARY_THRESHOLD']
    y_pred = model.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    pred3= pred2>binary_threshold
    pred4 = pred3*1
    return pred4, pred2


def neural_network(x_train,y_train,x_test,y_test):
    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
        print(variables)

    input_dim = variables['INPUT_DIM']
    batch_size = variables['BATCH_SIZE']
    epochs = variables['EPOCHS']

    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = Adam())
    
    model1= model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                      verbose=1, validation_data=(x_test, y_test))
    
    model.save(model_path+'new_model100.h5')
    saved_model = load_model(model_path+'new_model100.h5')
    
    plt.plot(model1.history['loss'])
    plt.plot(model1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(plot_path+'model_loss.png')    
    plt.show()
    

    y_pred, y_probs = prediction(saved_model, x_test)
    target_names = ['bad','good']
    
    print('=============NEURAL NETWORK PREDICTIONS==============')
    print(classification_report(y_test, y_pred, target_names=target_names))
    auc_score = roc_auc_score(y_test, y_probs)
    print ('AUC SCORE: ', round(auc_score, 2))



def best_model(model_name, x_test, y_test, i):
    with open('variables.yml') as f:
        variables = yaml.load(f, Loader=yaml.FullLoader)
        print(variables)
    binary_threshold = variables['BINARY_THRESHOLD']

    saved_model = load_model(model_name)
    y_pred = saved_model.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    pred3= pred2>binary_threshold
    pred4 = pred3*1
    target_names = ['bad','good']
    
    print('==================NEURAL NETWORK', i, 'PREDICTIONS============')    
    print(classification_report(y_test, pred4, target_names=target_names))
    auc_score = roc_auc_score(y_test, pred2)
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, pred2)
    return fpr, tpr, auc_score

def logistic(x_train,y_train,x_test,y_test):
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    yy = clf.predict(x_test)
    target_names = ['bad','good']
    
    print('================LOGISTIC REGRESSION PREDICTIONS===========')
    print(classification_report(y_test, yy, target_names=target_names))  
    auc_score = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score


def random_forest(x_train,y_train,x_test,y_test):    
    clf1 = RandomForestClassifier(max_depth=10, random_state=0)
    clf1.fit(x_train,y_train)
    target_names = ['bad','good']   
    yy= clf1.predict(x_test)
    
    print('==================RANDOM FOREST PREDICTIONS===============')
    print(classification_report(y_test, yy, target_names=target_names))
    auc_score = roc_auc_score(y_test, clf1.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, clf1.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score
    


def xgboost_model(x_train,y_train,x_test,y_test):  
    model = XGBClassifier()
    warnings.filterwarnings("ignore")
    model.fit(x_train,y_train) 
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    target_names = ['bad','good'] 

    print('=====================XGBOOST PREDICTIONS==================')
    print(classification_report(y_test, predictions, target_names=target_names))
    auc_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score


