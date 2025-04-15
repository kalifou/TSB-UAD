#!/bin/bash

path_to_dataset=../data/ESA-ADB/data/preprocessed/multivariate/
extension=/84_months.test.csv

for dataset in ESA-Mission1-semi-supervised ESA-Mission2-semi-supervised; do
    for algorith_i in DAMP LOC IForest; do
        echo $path_to_dataset$dataset$extension
        python -m TSB_UAD.esa_adb -m $algorith_i -pthd $path_to_dataset$dataset$extension -ch 15 --n-jobs 1 -tm
    done
done
