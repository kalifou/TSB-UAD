#!/bin/bash

#SBATCH --nodes=1                # node count
#SBATCH --cluster=hpda2 
#SBATCH --partition=hpda2_testgpu 
#hpda2_compute_gpu

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1 
#SBATCH --time=00:15:00
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --mem=70000

algorith_i=$1
dataset=$2
channel=$3

srun bash compute_node_terrabyte.sh $algorith_i $dataset $channel