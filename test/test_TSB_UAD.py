#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:21:04 2025

@author: trao_ka
"""

import unittest

from TSB_UAD import (get_arguments_esa_experiments,
                     get_channel_values_and_labels, 
                     run_AD_model,
                     LIST_AD_MODELS,
                     main
)

import pandas as pd

class TestAdESA(unittest.TestCase):

    def test_instanciation_of_models(self):

        for model_name in LIST_AD_MODELS:
            #assert model_name in LIST_AD_MODELS    
            
            path_to_esa_dataset = "data/ESA-ADB/data/preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv"
            
            channel_index_of_interrest = 15
            test_mode = True
            activate_plotting = False
            
            main(model_name, path_to_esa_dataset, channel_index_of_interrest, test_mode, activate_plotting)
            print("Passed the dry-running with: ", str(model_name))

if __name__ == "__main__":
    unittest.main()

    print("Every test has passed")
