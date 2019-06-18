#!/bin/bash


# BeiJing Air Quality [24 Hour Difference Model]

python ./train_bn_mcmc.py 21 0 1 0 SLP linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_SLP/ RMSE

python ./train_bn_mcmc.py 21 7 1 0 FFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_FFNN_RMSE/ RMSE

python ./train_bn_mcmc.py 21 7 1 0 LangevinFFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_LangevinFFNN/ RMSE

python ./train_bn_mcmc.py 21 7 1 3 DeepFFNN linear data/BeijingPM2.5/Train_set_24_hour_normalised.csv data/BeijingPM2.5/Test_set_24_hour_normalised.csv  results/BeijingPM2.5_v2_DeepFFNN_RMSE/ RMSE




python ./train_bn_mcmc.py 21 0 1 0 SLP sigmoid data/BeijingPM2.5/Train_set_24_hour_diff.csv data/BeijingPM2.5/Test_set_24_hour_diff.csv  results/BeijingPM2.5_SLP_Sigmoid_Diff/ RMSE



# ANALYSE THE RESULTS, GENERATE PLOTS AND TABLES

python analyse_test_results.py "./results/BeijingPM2.5_v2_SLP" "./results/BeijingPM2.5_v2_SLP/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config.yaml" 500 False TARGET_pm2.5_24_VALUE  pm2.5

python analyse_test_results.py "./results/BeijingPM2.5_v2_SLP_DIFF" "./results/BeijingPM2.5_v2_SLP_DIFF/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config_diff.yaml" 500 True TARGET_pm2.5_24_DIFF  pm2.5

python analyse_test_results.py "./results/BeijingPM2.5_SLP_Sigmoid_Diff" "./results/BeijingPM2.5_SLP_Sigmoid_Diff/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config_diff.yaml" 500 True TARGET_pm2.5_24_DIFF  pm2.5




python analyse_test_results.py "./results/BeijingPM2.5_v2_FFNN" "./results/BeijingPM2.5_v2_FFNN/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config.yaml" 500 False TARGET_pm2.5_24_VALUE pm2.5

python analyse_test_results.py "./results/BeijingPM2.5_v2_DeepFFNN_RMSE" "./results/BeijingPM2.5_v2_DeepFFNN_RMSE/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config.yaml" 500 False TARGET_pm2.5_24_VALUE pm2.5

python analyse_test_results.py "./results/BeijingPM2.5_v2_LangevinFFNN" "./results/BeijingPM2.5_v2_LangevinFFNN/test_predictions.tsv" "./data/BeijingPM2.5/Test_set_24_hour_full.csv" "data/BeijingPM2.5/Target_24_nzr_config.yaml" 500 False TARGET_pm2.5_24_VALUE pm2.5
