#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:21:04 2025

@author: trao_ka
"""

import unittest
import random

from TSB_UAD import (get_channel_values_and_labels, 
                     run_AD_model,
                     LIST_AD_MODELS,
                     main,
                     redistribute_anomalies_to_binary
)

class TestAdESA(unittest.TestCase):

    def test_instanciation_of_models(self):

        for model_name in LIST_AD_MODELS:
            #assert model_name in LIST_AD_MODELS    
            
            path_to_esa_dataset = "data/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv"
            path_to_save_logs = "results/tests/"
            
            channel_index_of_interrest = 15
            test_mode = True
            activate_plotting = False
            
            main(model_name, 
                 path_to_esa_dataset, 
                 path_to_save_logs,
                 channel_index_of_interrest, 
                 test_mode, 
                 activate_plotting,
                 n_jobs=1)
            
    
    def test_1_redistribute_to_binary(self):
        l = [0, 1, 2, 3, 4, 5, 0]
        gt = [0, 1, 1, 1, 1, 1, 0] 

        assert [redistribute_anomalies_to_binary(e) for e in l] == gt
        print("Test 1 completed!")

    def test_2_redistribute_to_binary(self):
        
        num_elements = random.randint(200, 1000)
        l = [random.randint(1, 19) in range(num_elements)]
        gt = [1 in range(num_elements)]

        assert [redistribute_anomalies_to_binary(e) for e in l] == gt
        print("Test 2 completed!")

    def test_3_redistribute_to_binary(self):
        
        num_elements = random.randint(200, 1000)
        l = [0 in range(num_elements)]
        gt = [0 in range(num_elements)]

        assert [redistribute_anomalies_to_binary(e) for e in l] == gt
        print("Test 2 completed!")

if __name__ == "__main__":
    unittest.main()

    print("Every test has passed")
