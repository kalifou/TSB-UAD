#!/bin/bash

channel=15


for dataset in ESA-Mission1-semi-supervised; do # ESA-Mission2-semi-supervised; do
    for algorith_i in DAMP LOF IForest MatrixProfile; do
        echo $algorith_i $dataset $channel
        sbatch login_node_terrabyte.sh $algorith_i $dataset $channel
    done
done

