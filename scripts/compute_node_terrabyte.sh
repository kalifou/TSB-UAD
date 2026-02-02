#!/bin/bash

##
# Defining the main function of the script to be run: includes the execution of the anomaly detection script.
##
main() {
    echo "Starting the execution of an algorithm.."
    source activate base
    #conda init bash
    conda activate TSB
    
    algorith_i=$1
    dataset=$2
    channel=$3
    local_log_file_name=$4
    
    path_to_dataset=data/ESA-ADB/binarized/multivariate/  #../data/ESA-ADB/data/preprocessed/multivariate/

    if [[ $dataset = "ESA-Mission1-semi-supervised" ]]
    then
      echo "It is dataset 1."
      extension=/84_months.test.csv #/42_months.train.csv
      results_path=../../MSAD/data/benchmark_esa_binarized/m1/test/
      ###
    elif [[ $dataset = "ESA-Mission2-semi-supervised" ]]
    then
      echo "it is Dataset 2."
      extension=/21_months.test.csv #/42_months.train.csv
      results_path=../../MSAD/data/benchmark_esa_binarized/m2/test/
      ##
    else
      echo "Unknown dataset."
    fi
    
    
    #results_path=../../MSAD/data/benchmark_esa_binarized/
    n_jobs=48
    
    python -m TSB_UAD.esa_adb -m $algorith_i -pthd $path_to_dataset$dataset$extension -ch $channel --n-jobs $n_jobs -pths $results_path >> $local_log_file_name
    
    echo "Done with algo: "$algorith_i >> $local_log_file_name
}

##
# Logging of the execution time needed to run the main function
##
log_file_name=$2"__"$1"__ch__"$3$(date +"__%Y_%m_%d__%H_%M_%S")"_.out"

(time main "$@" $log_file_name) &>> $log_file_name

