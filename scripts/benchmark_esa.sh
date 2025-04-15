#!/bin/bash

channel=15
path_to_dataset=../data/ESA-ADB/data/preprocessed/multivariate/
extension=/42_months.train.csv
results_path=../results/benchmark_esa/

for dataset in ESA-Mission1-semi-supervised; do # ESA-Mission2-semi-supervised; do
    for algorith_i in DAMP LOF IForest MatrixProfile; do
        echo $path_to_dataset$dataset$extension
        python -m TSB_UAD.esa_adb -m $algorith_i -pthd $path_to_dataset$dataset$extension -ch $channel --n-jobs 1 -tm -pths $results_path
    done
done
