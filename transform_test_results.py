#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import sys
import os

sys.path.append('../Dataset_Transformers')
from transform import Normalizer as nzr


#################################################################################
#
# TRANSFORM THE RESULTS OF TEST PREDICTIONS FOR A BAYESIAN MODEL 
#
# PARAMETERS
# - PATH TO WRITE RESULTING TRANSFORMED FILE
# - PATH TO TEST RESULT: THE PREDICTIONS MADE ON THE TEST DATA
# - PATH TO TESTING DATA: PATH TO THE ORIGINAL UN-NORMALISED DATA
# - PATH TO DE-NORMALISATION FILE: FILE CONTAINING PARAMTERS TO DE-NORMALISE TARGET PREDS
# - IS_NORMALISED: BOOLEAN
# - IS_DIFFERENCED: BOOLEAN
# - IS_PROPORTIONAL: BOOLEAN
# - ROUND: BOOLEAN
# - TARGET_COL_NAME: COLUMN NAME OF FOR PREDICTION TARGET
# - NAIVE_COL_NAME: FOR USE IF DIFFERENCING HAS BEEN APPLIED AND CALCULATING MASE
#
#################################################################################
def main():
    if len(sys.argv) < 10:
        print("ERROR: MISSING ARGUMENTS")
        print_usage(sys.argv)
        exit(1)
    else:
        result_path = sys.argv[1]
        result_file_path = sys.argv[2]
        test_data_path = sys.argv[3]
        norm_path = sys.argv[4]
        is_normalised = sys.argv[5]
        is_differenced = sys.argv[6]
        is_proportional = sys.argv[7]
        apply_round = sys.argv[8]
        target_col = sys.argv[9]
        ref_col = sys.argv[10]
        
        test_data = pd.read_csv( test_data_path, sep="," )
        test_preds = np.loadtxt( result_file_path )
        nzr_config = nzr.read_normalization_config( norm_path )

        final_preds = test_preds

        if is_normalised=='True':
            final_preds = nzr.de_normalize_all( test_preds, nzr_config )
  
        if is_differenced=='True':
            final_preds = de_difference( test_data, final_preds, ref_col, target_col )

        if is_proportional=='True':
            final_preds = de_prop_difference( test_data, final_preds, ref_col, target_col, apply_round )

        write_results(final_preds, result_path)


#################################################################################
# USAGE
#################################################################################
def print_usage(args):
    print("USAGE ")
    print(args[0], "<RESULTS DIR> <TEST PREDS FILE> <TEST DATA PATH> <DE NORM FILE>",
                   "  <IS NORMALISED> <IS DIFFERENCED> <IS PROPORTIONAL> <ROUND>",
                   "  <TARGET COL> <NAIVE COL>"
    )


#################################################################################
# OPEN THE NORMALISATION CONFIG FILE
#################################################################################
def load_normalisation_data(nzr_path):
    with open(nzr_path, 'r') as stream:
        lded = yaml.load(stream)
    return lded


#################################################################################
# DE-DIFFERENCE THE RAW PREDICTIONS
#################################################################################
def de_difference( data, preds, ref_col, target_col ):
    rez = preds.copy()
    for i in range(len(data)):
        rez[i,:] =  data.iloc[i,:][ref_col] + rez[i,:]
    return rez


#################################################################################
# DE-PROPORTIONAL DIFFERENCE THE RAW PREDICTIONS
#################################################################################
def de_prop_difference( data, preds, ref_col, target_col, apply_rounding ):
    rez = preds.copy()
    for i in range( len(data) ):
        rez[i,:] = data.iloc[i,:][ref_col] + ( rez[i,:] * data.iloc[i,:][ref_col] )
        if apply_rounding=='True':
            rez[i,:] =  np.around( rez[i,:] )
    return rez


#################################################################################
# WRITE OUT THE TRUE VALUES AND PREDICTION INTERVALS FILE
#################################################################################
def write_results( dataset, results_path ) :
    df = pd.DataFrame(dataset)
    df.to_csv(results_path, index=False, header=False)


if __name__ == "__main__": main()

