#!/bin/bash

python ./train_bn_mcmc.py 4 3 1 0 data/Lazer/train.txt, data/Lazer/test.txt results/Lazer_results/ 
python ./train_bn_mcmc.py 4 3 1 0 data/Sunspot/train.txt data/Sunspot/test.txt  results/Sunspot_results/
python ./train_bn_mcmc.py 4 3 1 0 data/Mackey/train.txt data/Mackey/test.txt  results/Mackey_results/
python ./train_bn_mcmc.py 4 3 1 0 data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_results/
python ./train_bn_mcmc.py 4 3 1 0 data/Henon/train.txt data/Henon/test.txt  results/Henon_results/
python ./train_bn_mcmc.py 4 3 1 0 data/Lorenz/train.txt data/Lorenz/test.txt  results/Lorenz_results/
python ./train_bn_mcmc.py 4 3 1 0 data/Rossler/train.txt data/Rossler/test.txt  results/Rossler_results/


python ./train_bn_mcmc.py 4 3 1 3 data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_deep_ffnn_results/
python ./train_bn_mcmc.py 4 3 1 1 data/ACFinance/train.txt data/ACFinance/test.txt  results/ACFinance_deep_ffnn_results/
