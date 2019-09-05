#!/bin/bash


python3 ./train_bn_mcmc.py 53 20 1 0 LangevinFFNN sigmoid data/Beijing/sets/Train_24_hour_norm.csv data/Beijing/sets/Test_24_hour_norm.csv results/Beijing_24Hr_LvnFFNN_Sigmoid/ MASE 100000

# PROCESS THE RESULTS SO THAT THEY ARE IN THE TARGET SPACE (DEAL WITH NORMALISED OR DIFFERENCED TARGETS)
 
python3 ./transform_test_results.py "./results/Beijing_24Hr_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./results/Beijing_24Hr_LvnFFNN_Sigmoid/test_predictions.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" "data/Beijing/sets/Target_24_nzr_config.yaml" True False False False "TARGET_pm2.5_24_VALUE" "pm2.5" 

python3 analyse_test_results.py "./results/Beijing_24Hr_LvnFFNN_Sigmoid" "./results/Beijing_24Hr_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" 50000 "TARGET_pm2.5_24_VALUE" "pm2.5"

