#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import sys
import os
import yaml

sys.path.append('../Dataset_Transformers')
from transform import Normalizer as nzr


#################################################################################
#
# ANALYSE THE RESULTS OF TEST PREDICTIONS FOR A BAYESIAN MODEL 
#
# WE WANT TO CREATE THE MEAN AND QUANTILE PREDICTIONS FROM THE RAW MODEL OUTPUTS
# AND THEN SUMMARISE THIS TO INDCATE HOW WELL CALIBRATED THE MODEL IS AND HOW
# WELL THE MEAN PERFORMS AS AN ESTIMATOR.
#
# PARAMETERS
# - PATH TO RESULTS: DIRECTORY IN WHICH TO WRITE RESULTS
# - PATH TO TEST RESULT: THE PREDICTIONS MADE ON THE TEST DATA
# - PATH TO TESTING DATA: PATH TO THE ORIGINAL UN-NORMALISED TEST DATA
# - BURNIN: INTEGER
# - TARGET_COL_NAME: COLUMN NAME OF FOR PREDICTION TARGET
# - NAIVE_COL_NAME: CALCULATING MASE
#
#################################################################################

def main():
    if len(sys.argv) < 6:
        print("ERROR: MISSING ARGUMENTS")
        print_usage(sys.argv)
        exit(1)
    else:
        result_path = sys.argv[1]
        result_file_path = sys.argv[2]
        test_data_path = sys.argv[3]
        burnin = sys.argv[4]
        target_col = sys.argv[5]
        ref_col = sys.argv[6]
        
        test_data = pd.read_csv(test_data_path, sep=",")
        test_preds = np.loadtxt(result_file_path)
        print( test_preds.shape )

        data_with_preds = add_mean_and_quantiles( test_data, test_preds, burnin )

        # NOW WE ARE READY TO PRODUCE SUMMARY AND PLOTS
        write_prediction_intervals_file( data_with_preds, target_col, result_path )
        summarise_model_calibration( data_with_preds, target_col, result_path )
        summarise_model_performance( data_with_preds, target_col, result_path, ref_col )


#################################################################################
# USAGE
#################################################################################
def print_usage(args):
    print("USAGE ")
    print(args[0], "<RESULTS DIR> <RESULTS FILE> <TEST DATA PATH> <BURNIN> <TARGET COL> <NAIVE COL>")



#################################################################################
# ADDING PREDICTINGS AND QUANTILE BANDS 
#################################################################################
def add_mean_and_quantiles( test_data, test_preds, burnin ) :
    rez = test_data.copy()
    tested = test_preds[int(burnin):, ]
    rez['mu'] = tested.mean(axis=0)
    rez['qnt_99'] = np.percentile(tested, 99, axis=0)
    rez['qnt_95'] = np.percentile(tested, 95, axis=0)
    rez['qnt_90'] = np.percentile(tested, 90, axis=0)
    rez['qnt_80'] = np.percentile(tested, 80, axis=0)
    rez['qnt_20'] = np.percentile(tested, 20, axis=0)
    rez['qnt_10'] = np.percentile(tested, 10, axis=0)
    rez['qnt_5'] = np.percentile(tested, 5, axis=0)
    rez['qnt_1'] = np.percentile(tested, 1, axis=0)     
    return rez


#################################################################################
# WRITE OUT THE TRUE VALUES AND PREDICTION INTERVALS FILE
#################################################################################
def write_prediction_intervals_file( test_data, target_field_name, results_path ) :

    data = {'y':test_data[target_field_name], 
            'qnt_1':test_data['qnt_1'], 
            'qnt_5':test_data['qnt_5'], 
            'qnt_10':test_data['qnt_10'], 
            'qnt_20':test_data['qnt_20'], 
            'mu': test_data['mu'], 
            'qnt_80':test_data['qnt_80'], 
            'qnt_90':test_data['qnt_90'], 
            'qnt_95':test_data['qnt_95'], 
            'qnt_99':test_data['qnt_99'] }
    df = pd.DataFrame(data)
    df.to_csv(results_path + '/testdata_prediction_intervals.csv', index=False)


#################################################################################
#  CALCULATE SUMMARY STATISTICS ABOUT THE CALIBRATION OF THE MODEL
#################################################################################
def summarise_model_calibration( test_data, target_name, results_path ):
    df = test_data.copy() 
    df["in_98_window"] = np.where( (df[target_name]>df['qnt_1']) & (df[target_name]<df['qnt_99']), 1, 0 )
    df["in_90_window"] = np.where( (df[target_name]>df['qnt_5']) & (df[target_name]<df['qnt_95']), 1, 0 )
    df["in_80_window"] = np.where( (df[target_name]>df['qnt_10']) & (df[target_name]<df['qnt_90']), 1, 0 )
    df["in_60_window"] = np.where( (df[target_name]>df['qnt_20']) & (df[target_name]<df['qnt_80']), 1, 0 )
    df["window_size_98"] =  df['qnt_99'] - df['qnt_1']
    df["window_size_90"] =  df['qnt_95'] - df['qnt_5']
    df["window_size_80"] =  df['qnt_90'] - df['qnt_10']
    df["window_size_60"] =  df['qnt_80'] - df['qnt_20']

    in_98_window = df["in_98_window"].mean()
    in_90_window = df["in_90_window"].mean()
    in_80_window = df["in_80_window"].mean()
    in_60_window = df["in_60_window"].mean()
    max_window_size_98 = df["window_size_98"].max()
    mean_window_size_98 = df["window_size_98"].mean()
    min_window_size_98 = df["window_size_98"].min()

    max_window_size_90 = df["window_size_90"].max()
    mean_window_size_90 = df["window_size_90"].mean()
    min_window_size_90 = df["window_size_90"].min()
    max_window_size_80 = df["window_size_80"].max()
    mean_window_size_80 = df["window_size_80"].mean()
    min_window_size_80 = df["window_size_80"].min()

    max_window_size_60 = df["window_size_60"].max()
    mean_window_size_60 = df["window_size_60"].mean()
    min_window_size_60 = df["window_size_60"].min()

    sum_data = { 'window': [98,90,80,60],
                 'calibration':[in_98_window,in_90_window,in_80_window,in_60_window],
                 'min_size':[min_window_size_98,min_window_size_90,min_window_size_80,min_window_size_60],
                 'mean_size':[mean_window_size_98,mean_window_size_90,mean_window_size_80,mean_window_size_60],
                 'max_size':[max_window_size_98,max_window_size_90,max_window_size_80,max_window_size_60] }
    sum_df = pd.DataFrame(sum_data)
    sum_df.to_csv(results_path + '/testdata_calibration.csv', index=False)


#################################################################################
#  CALCULATE PERFORMANCE STATISTICS OF THE MODEL
#################################################################################
def summarise_model_performance( test_data, target_name, results_path, naive_col='' ):
    df = test_data.copy()
    df["base_error"] =  df['mu'] - df[target_name]
    df["nominal_target"] = np.where(df[target_name]==0, 0.000001, df[target_name])
    df["abs_percent_error"] = abs(100*df["base_error"]/df["nominal_target"])
    df["abs_error"] =  abs(df["base_error"])
    df["naive_error"] =  df[naive_col] - df[target_name]
    df["abs_naive_error"] =  abs( df["naive_error"] )
    df["abs_percent_error_naive"] = abs(100*df["naive_error"]/df["nominal_target"])
    sum_data = {
                 'MAE': [ df["abs_error"].mean() ],
                 'MAPE': [ df["abs_percent_error"].mean() ],
                 'MASE': [ df["abs_error"].sum() / df["abs_naive_error"].sum() ],
                 'MAE Naive': [ df["abs_naive_error"].mean() ],
                 'MAPE Naive': [ df["abs_percent_error_naive"].mean() ]
               }
    sum_df = pd.DataFrame(sum_data)
    sum_df.to_csv(results_path + '/testdata_performance.csv', index=False)





if __name__ == "__main__": main()

