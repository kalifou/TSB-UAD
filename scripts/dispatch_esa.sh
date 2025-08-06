#!/bin/bash

#START=1
#END=76



for channel in $(seq 33 32 40 46 44 45 61 62 63); do
    for dataset in ESA-Mission1-semi-supervised; do # ESA-Mission2-semi-supervised; do
	# Cheaper algorithm evaluations - 7 hours
	for algorith_i_1 in DAMP IForest MatrixProfile SAND_offline PCA POLY CNN; do
            echo $algorith_i_1 $dataset $channel
            sbatch login_node_terrabyte_cheap.sh $algorith_i_1 $dataset $channel
	done
	# More expensive evaluations - 20 hours
	for algorith_i_2 in LOF SAND_online OCSVM LSTM; do                                         
            echo $algorith_i_2 $dataset $channel
            sbatch login_node_terrabyte_expensive.sh $algorith_i_2 $dataset $channel
        done
	
    done
done 
