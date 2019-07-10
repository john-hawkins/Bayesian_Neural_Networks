 
python transform_test_results.py "tests/test_01_result.csv" "tests/Test_01/test_predictions.tsv" "tests/Test_01/test_data.csv" "tests/Test_01/config.yaml" False False False False target_value current_value 

python transform_test_results.py "tests/test_02_result.csv" "tests/Test_02/test_predictions.tsv" "tests/Test_02/test_data.csv" "tests/Test_02/config.yaml" True False False False target_value current_value 

python transform_test_results.py "tests/test_03_result.csv" "tests/Test_03/test_predictions.tsv" "tests/Test_03/test_data.csv" "tests/Test_03/config.yaml" False True False False target_value current_value 

python transform_test_results.py "tests/test_04_result.csv" "tests/Test_04/test_predictions.tsv" "tests/Test_04/test_data.csv" "tests/Test_04/config.yaml" False False True True target_value current_value 


