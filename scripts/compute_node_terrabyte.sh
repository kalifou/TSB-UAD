#!/bin/bash

source activate base
#conda init bash
conda activate TSB

algorith_i=$1
dataset=$2
channel=$3

path_to_dataset=../data/ESA-ADB/data/preprocessed/multivariate/
extension=/42_months.train.csv
results_path=../results/benchmark_esa/
n_jobs=32

python -m TSB_UAD.esa_adb -m $algorith_i -pthd $path_to_dataset$dataset$extension -ch $channel --n-jobs $n_jobs -tm -pths $results_path  >> $algorith_i"__ch__"$channel$(date +"__%Y_%m_%d__%H_%M_%S")"_.out" 