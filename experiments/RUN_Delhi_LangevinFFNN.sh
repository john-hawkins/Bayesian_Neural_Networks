#!/bin/bash
  
cd ../

# Station 144
# Delhi Air Quality

python3 ./train_bn_mcmc.py 20 5 1 0 LangevinFFNN sigmoid data/Delhi/STN_144/train_normalised.csv  data/Delhi/STN_144/test_normalised.csv results/Delhi_144_LvnFFNN_Sigmoid/ MASE 100000

python3 ./transform_test_results.py "./results/Delhi_144_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./results/Delhi_144_LvnFFNN_Sigmoid/test_predictions.tsv" "./data/Delhi/STN_144/test.csv" "data/Delhi/STN_144/nzr_config.yaml" True False False False "TARGET_STN_144_PM10_7_VALUE" "STN_144_PM10"

python3 analyse_test_results.py "./results/Delhi_144_LvnFFNN_Sigmoid" "./results/Delhi_144_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./data/Delhi/STN_144/test.csv" 50000 "TARGET_STN_144_PM10_7_VALUE" "STN_144_PM10"


# Station 146
# Delhi Air Quality

python3 ./train_bn_mcmc.py 20 5 1 0 LangevinFFNN sigmoid data/Delhi/STN_146/train_normalised.csv  data/Delhi/STN_146/test_normalised.csv results/Delhi_146_LvnFFNN_Sigmoid/ MASE 100000

python3 ./transform_test_results.py "./results/Delhi_146_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./results/Delhi_146_LvnFFNN_Sigmoid/test_predictions.tsv" "./data/Delhi/STN_146/test.csv" "data/Delhi/STN_146/nzr_config.yaml" True False False False "TARGET_STN_146_PM10_7_VALUE" "STN_146_PM10"

python3 analyse_test_results.py "./results/Delhi_146_LvnFFNN_Sigmoid" "./results/Delhi_146_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./data/Delhi/STN_146/test.csv" 50000 "TARGET_STN_146_PM10_7_VALUE" "STN_146_PM10"




# Station 345
# Delhi Air Quality

python3 ./train_bn_mcmc.py 20 5 1 0 LangevinFFNN sigmoid data/Delhi/STN_345/train_normalised.csv  data/Delhi/STN_345/test_normalised.csv results/Delhi_345_LvnFFNN_Sigmoid/ MASE 100000

python3 ./transform_test_results.py "./results/Delhi_345_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./results/Delhi_345_LvnFFNN_Sigmoid/test_predictions.tsv" "./data/Delhi/STN_345/test.csv" "data/Delhi/STN_345/nzr_config.yaml" True False False False "TARGET_STN_345_PM10_7_VALUE" "STN_345_PM10"

python3 analyse_test_results.py "./results/Delhi_345_LvnFFNN_Sigmoid" "./results/Delhi_345_LvnFFNN_Sigmoid/test_predictions_final.tsv" "./data/Delhi/STN_345/test.csv" 50000 "TARGET_STN_345_PM10_7_VALUE" "STN_345_PM10"

