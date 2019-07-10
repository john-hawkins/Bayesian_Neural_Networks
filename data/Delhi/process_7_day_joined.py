import numpy as np
import pandas as pd

import sys
sys.path.append('../../../Dataset_Transformers')

from transform import DatasetGenerator as dg
from transform import Normalizer as nzr

df = pd.read_csv('delhi_pm10_all_stations_wide.csv')

index_column = "No"
forecast_column = "STN_144_PM10"
forecast_period = 7
list_of_lags = [7,14,21]

# NEED TO ADD THE INDEX COLUMN

df[index_column] = range( 0, len(df) )

new_df = dg.generate_time_dependent_features( df, index_column, forecast_column, forecast_period, list_of_lags)

cut_off_date = '2015-01-01'
train_df = new_df[ new_df['Date']<cut_off_date ]
test_df = new_df[ new_df['Date']>=cut_off_date ]

# ###########################################################################################################
# WRITE OUT THE FULL UN-NORMALISED VERSION
# ###########################################################################################################
train_df.to_csv('Station_144_Train.csv', encoding='utf-8', index=False, header=True)
test_df.to_csv('Station_144_Test.csv', encoding='utf-8', index=False, header=True)

# ###########################################################################################################
#  REMOVE UNWANTED COLUMNS, NORMALISE AND WRITE TO DISK
#  -- WE REMOVE THE DIFF VERSION OF THE TARGET 
#     AS IN THIS PROBLEM DATA IS GENERALLY STATIONARY (IT DOES NOT EXHIBIT OVERALL TREND)
# ###########################################################################################################
features = train_df.columns.tolist()
unwanted = ['No', 'Date', 'TARGET_STN_144_PM10_7_DIFF']

for x in unwanted : 
    features.remove(x)

train_df2 = train_df.loc[:,features]
test_df2 = test_df.loc[:,features]

target_col = "TARGET_STN_144_PM10_7_VALUE"

config = nzr.create_normalization_config(train_df2)

nzr.write_field_config(config, target_col, 'Delhi_Station_144__other_stns_nzr_config.yaml')

train_df_norm = nzr.normalize(train_df2, config, [])
test_df_norm = nzr.normalize(test_df2, config, [])

train_df_norm.to_csv('Station_144_others_Train_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('Station_144_others_Test_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)



