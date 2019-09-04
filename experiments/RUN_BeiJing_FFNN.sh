#!/bin/bash

cd ../

python ./train_bn_mcmc.py 53 20 1 0 FFNN sigmoid data/Beijing/sets/Train_24_hour_norm.csv data/Beijing/sets/Test_24_hour_norm.csv results/Beijing_FFNN_Sigmoid/  MASE 10000

python ./train_bn_mcmc.py 53 20 1 0 FFNN sigmoid data/Beijing/sets/Train_24_hour_diff.csv data/Beijing/sets/Test_24_hour_diff.csv  results/Beijing_FFNN_Sigmoid_Diff/  MASE 10000

python ./train_bn_mcmc.py 53 20 1 0 FFNN sigmoid data/Beijing/sets/Train_24_hour_prop_diff.csv data/Beijing/sets/Test_24_hour_prop_diff.csv results/Beijing_FFNN_Sigmoid_Prop_Diff/  MASE 10000


# PROCESS THE RESULTS SO THAT THEY ARE IN THE TARGET SPACE (DEAL WITH NORMALISED OR DIFFERENCED TARGETS)

python ./transform_test_results.py "./results/Beijing_FFNN_Sigmoid/test_predictions_final.tsv" "./results/Beijing_FFNN_Sigmoid/test_predictions.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" "data/Beijing/sets/Target_24_nzr_config.yaml" True False False False "TARGET_pm2.5_24_VALUE" "pm2.5"

python analyse_test_results.py "./results/Beijing_FFNN_Sigmoid" "./results/Beijing_FFNN_Sigmoid/test_predictions_final.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" 15000 "TARGET_pm2.5_24_VALUE" "pm2.5"

 
# Diff
 
python ./transform_test_results.py "./results/Beijing_FFNN_Sigmoid_Diff/test_predictions_final.tsv" "./results/Beijing_FFNN_Sigmoid_Diff/test_predictions.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" "data/Beijing/sets/Target_24_nzr_config_diff.yaml" True True False False "TARGET_pm2.5_24_DIFF" "pm2.5" 

python analyse_test_results.py "./results/Beijing_FFNN_Sigmoid_Diff" "./results/Beijing_FFNN_Sigmoid_Diff/test_predictions_final.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" 100 "TARGET_pm2.5_24_VALUE" "pm2.5"


# Proportional Diff

python ./transform_test_results.py "./results/Beijing_FFNN_Sigmoid_Prop_Diff/test_predictions_final.tsv" "./results/Beijing_FFNN_Sigmoid_Prop_Diff/test_predictions.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" "data/Beijing/sets/Target_24_nzr_config_prop_diff.yaml" True True True False "TARGET_pm2.5_24_PROP_DIFF" "pm2.5" 

python analyse_test_results.py "./results/Beijing_FFNN_Sigmoid_Prop_Diff" "./results/Beijing_FFNN_Sigmoid_Prop_Diff/test_predictions_final.tsv" "./data/Beijing/sets/Test_24_hour_full.csv" 100 "TARGET_pm2.5_24_VALUE" "pm2.5"


