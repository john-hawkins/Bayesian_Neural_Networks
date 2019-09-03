#!/bin/bash


python ./train_bn_mcmc.py 53 20 1 0 LangevinFFNN sigmoid data/Beijing/sets/Train_24_hour_norm.csv data/Beijing/sets/Test_24_hour_norm.csv results/Beijing_LvnFFNN_Sigmoid/ MASE 20000


# PROCESS THE RESULTS SO THAT THEY ARE IN THE TARGET SPACE (DEAL WITH NORMALISED OR DIFFERENCED TARGETS)
 
python ./transform_test_results.py "./results/Beijing_FFNN_Sigmoid_V2/test_predictions_final.tsv" "./results/Beijing_FFNN_Sigmoid_V2/test_predictions.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" "data/Beijing/sets/Target_24_nzr_config.yaml" True False False False "TARGET_pm2.5_24_VALUE" "pm2.5" 

python analyse_test_results.py "./results/Beijing_FFNN_Sigmoid_V2" "./results/Beijing_FFNN_Sigmoid_V2/test_predictions_final.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" 15000 "TARGET_pm2.5_24_VALUE" "pm2.5"

