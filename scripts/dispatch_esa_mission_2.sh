#!/bin/bash

#START=1
#END=76


# Iterate over all the Target Channels of Mission-2
for channel in $(seq 9 28)$" "$(seq 58 59)" "$(seq 70 91)" "$(seq 97 99); do
    for dataset in ESA-Mission2-semi-supervised; do # ESA-Mission2-semi-supervised; do
	# Cheaper algorithm evaluations - 7 hours
    	for algorith_i_1 in DAMP IForest MatrixProfile SAND_offline PCA POLY CNN; do #IForest
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