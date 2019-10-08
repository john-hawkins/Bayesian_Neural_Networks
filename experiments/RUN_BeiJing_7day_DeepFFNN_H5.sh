#!/bin/bash

cd ../

python3 ./train_bn_mcmc.py 28 5 1 3 DeepFFNN sigmoid data/Beijing/sets/Train_168_hour_norm.csv data/Beijing/sets/Test_168_hour_norm.csv results/Beijing_DeepFFNN_H5_7day_Sigmoid/  MASE 100000
# PROCESS THE RESULTS SO THAT THEY ARE IN THE TARGET SPACE (DEAL WITH NORMALISED OR DIFFERENCED TARGETS)

python3 ./transform_test_results.py "./results/Beijing_DeepFFNN_H5_7day_Sigmoid/test_predictions_final.tsv" "./results/Beijing_DeepFFNN_H5_7day_Sigmoid/test_predictions.tsv" "./data/Beijing/sets/Test_168_hour_full.csv" "data/Beijing/sets/Target_168_nzr_config.yaml" True False False False "TARGET_pm2.5_168_VALUE" "pm2.5"

python3 analyse_test_results.py "./results/Beijing_DeepFFNN_H5_7day_Sigmoid" "./results/Beijing_DeepFFNN_H5_7day_Sigmoid/test_predictions_final.tsv" "./data/Beijing/sets/Test_168_hour_full.csv" 50000 "TARGET_pm2.5_168_VALUE" "pm2.5"

