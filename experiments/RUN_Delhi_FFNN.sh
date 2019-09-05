#!/bin/bash
  
cd ../

# Delhi Air Quality

python3 ./train_bn_mcmc.py 20 5 1 0 FFNN sigmoid data/Delhi/STN_144/train_normalised.csv  data/Delhi/STN_144/test_normalised.csv results/Delhi_144_FFNN_Sigmoid/ MASE 100000 

python3 ./transform_test_results.py "./results/Delhi_144_FFNN_Sigmoid/test_predictions_final.tsv" "./results/Delhi_144_FFNN_Sigmoid/test_predictions.tsv" "./data/Delhi/STN_144/test.csv" "data/Delhi/STN_144/nzr_config.yaml" True False False False "TARGET_STN_144_PM10_7_VALUE" "STN_144_PM10"

python3 analyse_test_results.py "./results/Delhi_144_FFNN_Sigmoid" "./results/Delhi_144_FFNN_Sigmoid/test_predictions_final.tsv" "./data/Delhi/STN_144/test.csv" 50000 "TARGET_STN_144_PM10_7_VALUE" "STN_144_PM10"


