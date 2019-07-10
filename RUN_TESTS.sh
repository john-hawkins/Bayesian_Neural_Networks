#!/bin/bash

# THESE SIMPLE TESTS ENSURE THAT THE RESULTS MODIFICATION SCRIPTS FUNCTION AS EXPECTED

python transform_test_results.py "tests/test_01_result.csv" "tests/Test_01/test_predictions.tsv" "tests/Test_01/test_data.csv" "tests/Test_01/config.yaml" False False False False target_value current_value 

python transform_test_results.py "tests/test_02_result.csv" "tests/Test_02/test_predictions.tsv" "tests/Test_02/test_data.csv" "tests/Test_02/config.yaml" True False False False target_value current_value 

python transform_test_results.py "tests/test_03_result.csv" "tests/Test_03/test_predictions.tsv" "tests/Test_03/test_data.csv" "tests/Test_03/config.yaml" False True False False target_value current_value 

python transform_test_results.py "tests/test_04_result.csv" "tests/Test_04/test_predictions.tsv" "tests/Test_04/test_data.csv" "tests/Test_04/config.yaml" False False True True target_value current_value 

FAILED="False"
DIFF="$(diff "./tests/test_01_result.csv" "./tests/test_02_result.csv")"
if [ "$DIFF" != "" ] 
then
    echo "TEST 2 LIKELY FAILED"
    FAILED='True'
fi

DIFF="$(diff "./tests/test_01_result.csv" "./tests/test_03_result.csv")"
if [ "$DIFF" != "" ]
then
    echo "TEST 3 LIKELY FAILED"
    FAILED='True'
fi

DIFF="$(diff "./tests/test_01_result.csv" "./tests/test_04_result.csv")"
if [ "$DIFF" != "" ]
then
    echo "TEST 4 LIKELY FAILED"
    FAILED='True'
fi

if [ "$FAILED" == "False" ]
then
    echo "ALL TESTS PASSED"
fi

rm tests/test_01_result.csv
rm tests/test_02_result.csv
rm tests/test_03_result.csv
rm tests/test_04_result.csv

