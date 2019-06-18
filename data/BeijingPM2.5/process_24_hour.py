import numpy as np
import pandas as pd

import sys
sys.path.append('../../../Dataset_Transformers')

from transform import DatasetGenerator as dg
from transform import Normalizer as nzr

df = pd.read_csv('data.csv')
df2 = df[24:].copy()
df2['N'] = np.where(df2.cbwd.str[0:1]=='N', 1, 0)
df2['S'] = np.where(df2.cbwd.str[0:1]=='S', 1, 0)
df2['E'] = np.where(df2.cbwd.str[1:2]=='E', 1, 0)
df2['W'] = np.where(df2.cbwd.str[1:2]=='W', 1, 0)
df2.drop(["cbwd"],axis = 1, inplace = True) 

index_column = "No"
forecast_column = "pm2.5"
forecast_period = 24
list_of_lags = [1,2,24,48]

new_df = dg.generate_time_dependent_features( df2, index_column, forecast_column, forecast_period, list_of_lags )

trainset = 30000
train_df = new_df.loc[0:trainset,:]
test_df = new_df.loc[trainset+1:,:]

# ###########################################################################################################
# WRITE OUT THE FULL UN-NORMALISED VERSION
# ###########################################################################################################
train_df.to_csv('Train_set_24_hour_full.csv', sep=' ', encoding='utf-8', index=False, header=True)
test_df.to_csv('Test_set_24_hour_full.csv', sep=' ', encoding='utf-8', index=False, header=True)

# ###########################################################################################################
#  REMOVE UNWANTED COLUMNS, NORMALISE AND WRITE TO DISK
# ###########################################################################################################
features = train_df.columns.tolist()
unwanted = ["No", "year", "month", "day", "TARGET_pm2.5_24_DIFF"]
for x in unwanted : 
    features.remove(x)

train_df2 = train_df.loc[:,features]
test_df2 = test_df.loc[:,features]

target_col = "TARGET_pm2.5_24_VALUE"

config = nzr.create_padded_normalization_config(train_df2, 0.05)
#config = nzr.create_normalization_config(train_df2)
 
nzr.write_field_config(config, target_col, 'Target_24_nzr_config.yaml')

train_df_norm = nzr.normalize(train_df2, config, ['N','S','E','W'])
test_df_norm = nzr.normalize(test_df2, config, ['N','S','E','W'])

train_df_norm.to_csv('Train_set_24_hour_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('Test_set_24_hour_normalised.csv', sep=' ', encoding='utf-8', index=False, header=False)



# ###########################################################################################################
#  DIFFERENCED VERSION -  REMOVE UNWANTED COLUMNS, NORMALISE AND WRITE TO DISK
# ###########################################################################################################
features = train_df.columns.tolist()
unwanted = ["No", "year", "month", "day", 'TARGET_pm2.5_24_VALUE' ]
for x in unwanted : 
    features.remove(x)

train_df2 = train_df.loc[:,features]
test_df2 = test_df.loc[:,features]

target_col = "TARGET_pm2.5_24_DIFF"


config = nzr.create_padded_normalization_config(train_df2, 0.05)
# config = nzr.create_normalization_config(train_df2)
 
nzr.write_field_config(config, target_col, 'Target_24_nzr_config_diff.yaml')

train_df_norm = nzr.normalize(train_df2, config, ['N','S','E','W'])
test_df_norm = nzr.normalize(test_df2, config, ['N','S','E','W'])

train_df_norm.to_csv('Train_set_24_hour_diff.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('Test_set_24_hour_diff.csv', sep=' ', encoding='utf-8', index=False, header=False)

