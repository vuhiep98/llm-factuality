#!/bin/bash
#PBS -q GPU-1A
#PBS -N ChatGPT

source ~/miniconda3/bin/activate m-factscore
cd $PBS_O_WORKDIR
bash scripts/factscore_llama.sh 0 ChatGPT