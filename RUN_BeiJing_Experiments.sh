#!/bin/bash


# BeiJing Air Quality [24 Hour Difference Model]

python ./train_bn_mcmc.py 22 0 1 0 SLP linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_SLP/ RMSE

python ./train_bn_mcmc.py 22 7 1 0 FFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_FFNN_RMSE/ RMSE

python ./train_bn_mcmc.py 22 7 1 0 LangevinFFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_LangevinFFNN/ RMSE

python ./train_bn_mcmc.py 22 7 1 3 DeepFFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_DeepFFNN_RMSE/ RMSE

