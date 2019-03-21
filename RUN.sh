#!/bin/bash

# SIMPLE TEST DATA

python ./train_bn_mcmc.py 4 0 1 0 SLP sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_SLP/ MASE

python ./train_bn_mcmc.py 4 3 1 0 FFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_FFNN/ MASE

python ./train_bn_mcmc.py 4 3 1 0 LangevinFFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_LangevinFFNN/ MASE

python ./train_bn_mcmc.py 4 3 1 3 DeepFFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_Deep_FFNN/ MASE

