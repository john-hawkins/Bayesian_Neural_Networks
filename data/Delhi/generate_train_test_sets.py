import numpy as np
import pandas as pd
import sys
import os

sys.path.append('../../../Dataset_Transformers')

from transform import DatasetGenerator as dg
from transform import Normalizer as nzr

df = pd.read_csv('delhi_pm10_all_stations_wide.csv')

index_column = "No"
forecast_period = 7
list_of_lags = [7,14,21]

cut_off_date = '2015-01-01'

# NEED TO ADD THE INDEX COLUMN
df[index_column] = range( 0, len(df) )

# FUNCTIONS
#################################################################################
def ensure_dir(results_dir):
    directory = os.path.abspath(results_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_files_for_station(stat, index_column, cut_off_date, forecast_period, list_of_lags):
    folder = 'STN_%s' % str(stat)
    ensure_dir(folder)
    forecast_column = 'STN_%s_PM10' % str(stat)
    new_df = dg.generate_time_dependent_features( df, index_column, forecast_column, forecast_period, list_of_lags)
    train_df = new_df[ new_df['Date']<cut_off_date ]
    test_df = new_df[ new_df['Date']>=cut_off_date ]
    # ###########################################################################################################
    #  REMOVE UNWANTED COLUMNS, NORMALISE AND WRITE TO DISK
    #  -- WE REMOVE THE DIFF VERSION OF THE TARGET 
    # AS IN THIS PROBLEM DATA IS GENERALLY STATIONARY (IT DOES NOT EXHIBIT OVERALL TREND)
    # ###########################################################################################################
    features = train_df.columns.tolist()

    val_name = 'STN_%s_PM10' % str(stat)
    targ1_name = 'TARGET_STN_%s_PM10_7_VALUE' % str(stat)
    targ2_name = 'TARGET_STN_%s_PM10_7_DIFF' % str(stat)
    targ3_name = 'TARGET_STN_%s_PM10_7_PROP_DIFF' % str(stat)
    unwanted = ['No', 'Date', val_name, targ1_name, targ2_name, targ3_name]

    for x in unwanted : 
        features.remove(x)
    features.append(val_name)
    features.append(targ1_name)
    # WRITE OUT THE UN-NORMALISED VERSION
    train_df2 = train_df.loc[:, features]
    test_df2 = test_df.loc[:, features] 
    train_df2.to_csv(folder+'/train.csv', encoding='utf-8', index=False, header=True)
    test_df2.to_csv(folder+'/test.csv', encoding='utf-8', index=False, header=True)

    config = nzr.create_normalization_config(train_df2)
    nzr.write_field_config(config, targ1_name, folder+'/nzr_config.yaml')

    train_df_norm = nzr.normalize(train_df2, config, [])
    test_df_norm = nzr.normalize(test_df2, config, [])

    train_df_norm.to_csv(folder+'/train_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)
    test_df_norm.to_csv(folder+'/test_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)



generate_files_for_station(144, index_column, cut_off_date, forecast_period, list_of_lags)
generate_files_for_station(146, index_column, cut_off_date, forecast_period, list_of_lags)
generate_files_for_station(345, index_column, cut_off_date, forecast_period, list_of_lags)


