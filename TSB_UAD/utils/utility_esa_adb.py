#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:16:38 2025

@author: trao_ka
"""

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

import numpy as np
import math

from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length

from sklearn.preprocessing import MinMaxScaler


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
    
    score = clf.decision_scores_
    
    if model_name == 'OCSVM':
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    
    if model_name in ['DAMP', 'IForest', 'LOF', 'MatrixProfile','PCA']:
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        
    return score, slidingWindow


def get_channel_values_and_labels(channel_index, dataframe):
    return dataframe['channel_' + str(channel_index)].values, dataframe['is_anomaly_channel_' + str(channel_index)].values

def get_arguments_esa_experiments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="Select the AD model to use among: "
        + "{'DAMP', 'SAND (offline)', 'SAND (online)','IForest', 'LOF', 'MatrixProfile', 'PCA', 'POLY', 'OCSVM', 'LSTM', 'CNN'}.", 
        type=str,
        required=True
    )
    
    parser.add_argument(
        "-ch",
        "--channel-number",
        help="Select the Telemmetry channel number to use (univariate analysis). Example - Channel 15", 
        type=int,
        required=True
    )
    parser.add_argument(
        "-pthd",
        "--path-to-esa-dataset",
        help="Provide with the path to the ESA-ADB dataset split of interest . Example for mission 1: {3_months.train.csv, 42_months.train.csv, 84_months.train.csv, ..}", 
        type=str,
        required=False,
        default="../../data/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv"
    )
    
    parser.add_argument(
        "-tm",
        "--activate-test-mode",
        help="Dry run of the model on a subset of your dataset and channel.", 
        action="store_true",
        default=False,
    )
    
    parser.add_argument(
        "-plt",
        "--activate-plotting",
        help="Activate the plotting functionality for analyzing the results.", 
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()
    
    model_name = args.model
    path_to_esa_dataset = args.path_to_esa_dataset
    channel_index_of_interrest = args.channel_number
    test_mode = args.activate_test_mode
    activate_plotting = args.activate_plotting
    
    return model_name, path_to_esa_dataset, channel_index_of_interrest, test_mode, activate_plotting