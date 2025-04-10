#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:54:56 2025

@author: trao_ka
"""

import warnings
warnings.filterwarnings('ignore')

from TSB_UAD.utils.visualisation import plotFig
import matplotlib.pyplot as plt
import pandas as pd
from TSB_UAD.vus.metrics import get_metrics

from TSB_UAD.utils.utility_esa_adb import get_channel_values_and_labels, get_arguments_esa_experiments, run_AD_model



def main(model_name, 
         path_to_esa_dataset, 
         channel_index_of_interrest, 
         test_mode,
         activate_plot):
    
    df_esa_dataset = pd.read_csv(path_to_esa_dataset)
    
    m1_values, m1_labels = get_channel_values_and_labels(channel_index=channel_index_of_interrest, 
                                                         dataframe=df_esa_dataset)
    
    name = path_to_esa_dataset.split('/')[-2] + "-" + path_to_esa_dataset.split('/')[-1] + "- ch -" + str(channel_index_of_interrest)
        
    max_length = 50000000

    if test_mode:
        max_length = 5000
    
    data = m1_values[:max_length].astype(float)
    label = m1_labels[:max_length].astype(int)
    
    score_local, slidingWindow_local = run_AD_model(data, label, model_name)
    plotFig(data, label, score_local, slidingWindow_local, fileName=name, modelName=model_name)
    
    if activate_plot:
        plt.show()
        
    #Print accuracy
    #results = get_metrics(score_local, label, metric="all", slidingWindow=slidingWindow_local)
    #for metric in results.keys():
    #    print(metric, ':', results[metric])
    
    pass

if __name__ == "__main__":

    model_name, path_to_esa_dataset, channel_index_of_interrest, test_mode, activate_plotting = get_arguments_esa_experiments()
    
    main(model_name, path_to_esa_dataset, channel_index_of_interrest, test_mode, activate_plotting)
