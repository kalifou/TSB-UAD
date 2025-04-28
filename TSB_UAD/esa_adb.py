#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:54:56 2025

@author: trao_ka
"""

import ipdb
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from TSB_UAD.utils.visualisation import plotFig
import matplotlib.pyplot as plt
import pandas as pd
from TSB_UAD.vus.metrics import get_metrics

from TSB_UAD.utils.utility_esa_adb import get_channel_values_and_labels, get_arguments_esa_experiments, run_AD_model



def main(model_name, 
         path_to_esa_dataset, 
         path_to_save_logs,
         channel_index_of_interrest, 
         test_mode,
         activate_plot, 
         n_jobs=1):
    
    df_esa_dataset = pd.read_csv(path_to_esa_dataset)
    
    m1_values, m1_labels = get_channel_values_and_labels(channel_index=channel_index_of_interrest, 
                                                         dataframe=df_esa_dataset)
    
    name_to_dataset_split = path_to_esa_dataset.split('/')[-2] + "-" + path_to_esa_dataset.split('/')[-1].split('.')[0]
    whole_name_experiments = name_to_dataset_split + "-ch-" + str(channel_index_of_interrest)
    max_length = 50000000

    if test_mode:
        max_length = 5000
    
    data = m1_values[:max_length].astype(float)
    label = m1_labels[:max_length].astype(int)
    
    score_local, slidingWindow_local = run_AD_model(data, 
                                                    label, 
                                                    model_name, 
                                                    n_jobs=n_jobs)
    
    if activate_plot:
        plotFig(data, label, score_local, slidingWindow_local, fileName=whole_name_experiments, modelName=model_name)
        plt.show()
    
    metrics_dir_local = path_to_save_logs + "/metrics/"  + model_name + "/"
    scores_dir_local = path_to_save_logs + "/scores/" + name_to_dataset_split + "/" + model_name + "/score/"
    score_filename = scores_dir_local + "/"+ whole_name_experiments + ".out"
    
    # Logging of the scores
    if not os.path.exists(scores_dir_local):
         os.makedirs(scores_dir_local)
        
    np.savetxt(score_filename, score_local, delimiter=",")
    
    # Logging of the metrics
    if not os.path.exists(metrics_dir_local):
        os.makedirs(metrics_dir_local)
    
    if not test_mode:
        vus_i = get_metrics(score=score_local, labels=label, metric="vus", slidingWindow=slidingWindow_local)
        range_auc_i = get_metrics(score=score_local, labels=label, metric="range_auc", slidingWindow=slidingWindow_local)
        print("Model :", model_name, vus_i, range_auc_i)
         
    
    pass

if __name__ == "__main__":

    model_name, path_to_esa_dataset, path_to_save_logs, channel_index_of_interrest, test_mode, activate_plotting, n_jobs = get_arguments_esa_experiments()
    
    main(model_name, path_to_esa_dataset, path_to_save_logs, channel_index_of_interrest, test_mode, activate_plotting, n_jobs=n_jobs)
