#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:54:56 2025

@author: trao_ka
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.utils.visualisation import plotFig
from sklearn.preprocessing import MinMaxScaler

#from TSB_UAD.models.norma import NORMA
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.matrix_profile import MatrixProfile
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.poly import POLY
from TSB_UAD.models.ocsvm import OCSVM
from TSB_UAD.models.lstm import lstm
#from TSB_UAD.models.AE_mlp2 import AE_MLP2
from TSB_UAD.models.cnn import cnn
#from TSB_UAD.models.series2graph import Series2Graph

from TSB_UAD.models.damp import DAMP
from TSB_UAD.models.sand import SAND

from TSB_UAD.vus.metrics import get_metrics

import argparse

LIST_AD_MODELS = ['DAMP', 'SAND (offline)', 'SAND (online)',
                  'IForest', 'LOF', 'MatrixProfile', 'PCA', 
                  'POLY', 'OCSVM', 'LSTM', 'CNN'
                  ]

def run_AD_model(data, label, model_name):
    
    assert model_name in LIST_AD_MODELS
    
    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()
    
    
    # Prepare data for semisupervised method. 
    # Here, the training ratio = 0.1
    
    data_train = data[:int(0.1*len(data))]
    data_test = data
    
    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    
    
    if model_name == 'DAMP':
    
        clf = DAMP(m = slidingWindow,sp_index=slidingWindow+1)
        x = data
        clf.fit(x)
        
        
    elif model_name == 'SAND (offline)':
    
    
        clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
        x = data
        clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
    
    elif model_name == 'SAND (online)':
    
        clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
        x = data
        clf.fit(x,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
    
    elif model_name == 'IForest':
    
        clf = IForest(n_jobs=1)
        x = X_data
        clf.fit(x)
    
    elif model_name == 'LOF':
    
        clf = LOF(n_neighbors=20, n_jobs=1)
        x = X_data
        clf.fit(x)
    
    elif model_name == 'MatrixProfile':
    
        clf = MatrixProfile(window = slidingWindow)
        x = data
        clf.fit(x)
    
    elif model_name == 'PCA':
        clf = PCA()
        x = X_data
        clf.fit(x)
    
    
    elif model_name == 'POLY':
        clf = POLY(power=3, window = slidingWindow)
        x = data
        clf.fit(x)
    
    
    elif model_name =='OCSVM':
        X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
        X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
    
        clf = OCSVM(nu=0.05)
        clf.fit(X_train_, X_test_)

    
    elif model_name == 'LSTM':
        clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 50, patience = 5, verbose=0)
        clf.fit(data_train, data_test)
    
    
    elif model_name == 'CNN':
        clf = cnn(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 100, patience = 5, verbose=0)
    
        clf.fit(data_train, data_test)


    if model_name in ["LSTM", "CNN", "POLY"]:
        # Post training: evaluating the model and plotting the results
        measure = Fourier()    
        measure.detector = clf
        measure.set_param()
        clf.decision_function(measure=measure)
    
    if model_name == 'MatrixProfile':
    
        clf.score(query_length=2*slidingWindow,dataset=x)
        score = clf.score
    else:   
        score = clf.decision_scores_
    
    if model_name == 'OCSVM':
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    
    if model_name in ['DAMP', 'IForest', 'LOF', 'MatrixProfile','PCA']:
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        
    return score, slidingWindow


def get_channel_values_and_labels(channel_index, dataframe):
    return dataframe['channel_' + str(channel_index)].values, dataframe['is_anomaly_channel_' + str(channel_index)].values


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="Select the AD model to use among: "
        + "{'DAMP', 'SAND (offline)', 'SAND (online)','IForest', 'LOF', 'MatrixProfile', 'PCA', 'POLY', 'OCSVM', 'LSTM', 'CNN'}.", 
        type=str
    )
    args = parser.parse_args()
    model_name = args.model
    
    path_esa_mission_1 = "/localhome/trao_ka/Documents/projects/hai_vouchers/"
    path_esa_mission_1 += "anomaly_detection_gallileo/esa_extension/ESA-ADB/data/preprocessed/multivariate/"
    path_esa_mission_1 += "ESA-Mission1-semi-supervised/3_months.train.csv"
    
    df_esa_mission_1_train = pd.read_csv(path_esa_mission_1)
    channel_index_of_interrest = 15
    
    m1_values, m1_labels = get_channel_values_and_labels(channel_index=channel_index_of_interrest, 
                                                         dataframe=df_esa_mission_1_train)
    # Prepare data for unsupervised method

    #filepath = '../../data/benchmark/ECG/MBA_ECG805_data.out'
    #df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    
    #name = filepath.split('/')[-1]
    name = "ESA-ADB-MISSION-1-TRAIN"
    max_length = 10000000
    
    data = m1_values[:max_length].astype(float)
    label = m1_labels[:max_length].astype(int)
    
    score_local, slidingWindow_local = run_AD_model(data, label, model_name)
    plotFig(data, label, score_local, slidingWindow_local, fileName=name, modelName=model_name)
    
    
    #Print accuracy
    results = get_metrics(score_local, label, metric="all", slidingWindow=slidingWindow_local)
    for metric in results.keys():
        print(metric, ':', results[metric])