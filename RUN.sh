#!/bin/bash

python ./train_bn_mcmc.py 4 0 1 0 SLP sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_SLP/ MASE

python ./train_bn_mcmc.py 4 3 1 0 FFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_FFNN/ MASE

python ./train_bn_mcmc.py 4 3 1 0 LangevinFFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_LangevinFFNN/ MASE

python ./train_bn_mcmc.py 4 3 1 3 DeepFFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_Deep_FFNN/ MASE


# HIGGS BOSON DATASET
python ./train_bn_mcmc.py 28 0 1 0 SLP sigmoid data/HIGGS/train_small.csv data/HIGGS/test.csv  results/HIGGS_SLP/ AUC

python ./train_bn_mcmc.py 28 14 1 0 FFNN sigmoid data/HIGGS/train_small.csv data/HIGGS/test.csv  results/HIGGS_FFNN/ AUC

python ./train_bn_mcmc.py 28 7 1 5 DeepFFNN sigmoid data/HIGGS/train_small.csv data/HIGGS/test.csv  results/HIGGS_DeepFFNN/ AUC

python ./train_bn_mcmc.py 28 7 1 5 LangevinFFNN sigmoid data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_LangevinFFNN/ MASE


# BeiJing Air Quality 
python ./train_bn_mcmc.py 13 0 1 0 SLP linear data/BeijingPM2.5/train_set_norm.csv data/BeijingPM2.5/test_set_norm.csv  results/BeijingPM2.5_SLP/ MASE

python ./train_bn_mcmc.py 13 7 1 0 FFNN linear data/BeijingPM2.5/train_set_norm.csv data/BeijingPM2.5/test_set_norm.csv  results/BeijingPM2.5_FFNN/ MASE

python ./train_bn_mcmc.py 13 7 1 5 DeepFFNN linear data/BeijingPM2.5/train_set_norm.csv data/BeijingPM2.5/test_set_norm.csv  results/BeijingPM2.5_DeepFFNN/ MASE



