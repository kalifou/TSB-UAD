#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:57:50 2024

@author: trao_ka
"""

import pandas as pd
import ipdb
import argparse

import os

from TSB_UAD.utils.utility_esa_adb import redistribute_anomalies_to_binary, \
    get_anomaly_labels




def transform_chunck_i(pandas_data_chunck_i, index_i, all_labels, anomaly_labels):

    # reading the column labels once, in order to retrieve the column names associated with anomaly gt.
    if index_i == 0:
        all_labels = pandas_data_chunck_i.columns.tolist()
        anomaly_labels = get_anomaly_labels(all_labels)
    

    # Converting the anomaly labels from 4 classes to a binary class problem 
    for label_i in anomaly_labels:
        if label_i != None:
            labels = pandas_data_chunck_i[label_i].apply(lambda x: redistribute_anomalies_to_binary(x))
            pandas_data_chunck_i[label_i] = labels
            pandas_data_chunck_i[label_i] =  pandas_data_chunck_i[label_i].astype(int)
            
    return pandas_data_chunck_i, all_labels, anomaly_labels




    
def generate_and_save_binarized_esa_adb_dataset(path_read_data, 
                                                path_write_data, 
                                                chunck_size, 
                                                test_mode=False):

    local_data = pd.read_csv(path_read_data, chunksize=chunck_size)

    cpt=0
    header = True
    anomaly_labels = None
    only_anomaly_labels = None
    
    for chunck_i in local_data:
        
        chunck_i, anomaly_labels, only_anomaly_labels = transform_chunck_i(pandas_data_chunck_i=chunck_i, 
                                                                          index_i=cpt, 
                                                                          all_labels=anomaly_labels, 
                                                                          anomaly_labels=only_anomaly_labels,
                                                                          )
        if cpt >0:
            header = False
                
        chunck_i.to_csv(path_write_data, mode='a', header=header)
        
        if cpt >=5 and test_mode is True:
            break
        cpt +=1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-pthd", "--path-to-dataset", 
                        help="Provide with the absolute path to the dataset.",
                        type=str,
                        default="data/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv")
    
    parser.add_argument("-pths", "--path-to-save-binarized", 
                        help="Provide with the absolute path to the location where to save the fully binarized dataset",
                        type=str,
                        default="/tmp/")
    
    parser.add_argument("-tm", "--test-mode", action="store_true", 
                        help="Activate the test mode.") 
    
    args = parser.parse_args()
    print(args)
    
    path_to_dataset = args.path_to_dataset
    path_to_save_binarized = args.path_to_save_binarized
    test_mode = args.test_mode
    
    assert len(path_to_dataset) > 0    
    chunck_size = 50000
    
    # Logging of the metrics
    if not os.path.exists(path_to_save_binarized):
        os.makedirs(path_to_save_binarized)
    
    file_name = path_to_dataset.split("/")[-1]
    generate_and_save_binarized_esa_adb_dataset(path_read_data=path_to_dataset, 
                                                path_write_data=path_to_save_binarized + "/" + file_name, 
                                                chunck_size=chunck_size,
                                                test_mode=test_mode)
