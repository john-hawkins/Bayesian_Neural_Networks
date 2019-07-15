#!/bin/bash

# THESE SIMPLE TESTS ENSURE THAT THE ANALYSIS OF RESULTS SCRIPTS FUNCTION AS EXPECTED


python analyse_test_results.py  "tests/Test_01/" "tests/Test_01/test_predictions.tsv" "tests/Test_01/test_data.csv" 2 target_value current_value

python analyse_test_results.py  "tests/Test_05/" "tests/Test_05/test_predictions.tsv" "tests/Test_05/test_data.csv" 2 target_value current_value

python analyse_test_results.py  "tests/Test_06/" "tests/Test_06/test_predictions.tsv" "tests/Test_06/test_data.csv" 2 target_value current_value

python analyse_test_results.py  "tests/Test_07/" "tests/Test_07/test_predictions.tsv" "tests/Test_07/test_data.csv" 2 target_value current_value

FAILED="False"

DIFF="$(diff "./tests/Test_01/testdata_performance.csv" "./tests/Test_01/benchmark.csv")"
if [ "$DIFF" != "" ] 
then
    echo "TEST 1 FAILED"
    FAILED='True'
fi


DIFF="$(diff "./tests/Test_05/testdata_performance.csv" "./tests/Test_05/benchmark.csv")"
if [ "$DIFF" != "" ]
then
    echo "TEST 5 FAILED"
    FAILED='True'
fi

DIFF="$(diff "./tests/Test_06/testdata_performance.csv" "./tests/Test_06/benchmark.csv")"
if [ "$DIFF" != "" ]
then
    echo "TEST 6 FAILED"
    FAILED='True'
fi

DIFF="$(diff "./tests/Test_07/testdata_performance.csv" "./tests/Test_07/benchmark.csv")"
if [ "$DIFF" != "" ]
then
    echo "TEST 7 FAILED"
    FAILED='True'
fi


if [ "$FAILED" == "False" ]
then
    echo "ALL ANALYSE TESTS PASSED"
fi
 

