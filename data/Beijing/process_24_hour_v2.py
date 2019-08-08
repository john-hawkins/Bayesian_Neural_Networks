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

# WHERE PM2.5 IS ZERO - POTENTIAL MEASUREMENT LIMIT ERROR - REPLACE WITH NOMINAL SMALL VALUE
default_value = 0.01
df2["pm2.5"] = np.where(df2["pm2.5"] == 0, default_value,df2["pm2.5"] )

index_column = "No"
forecast_column = "pm2.5"
forecast_period = 24
list_of_lags = [1,2,3,4,5,6,24,48,72,168]
list_of_mas = [6,24,168]

new_df = dg.generate_time_dependent_features( df2, index_column, forecast_column, forecast_period, list_of_lags, list_of_mas  )
features = new_df.columns.tolist()
final_two = [ "pm2.5", "TARGET_pm2.5_24_VALUE" ]
for x in final_two :
    features.remove(x)

features.extend(final_two)
new_df = new_df[features]

trainset = 30000
train_df = new_df.loc[0:trainset,:]
test_df = new_df.loc[trainset+1:,:]

# ###########################################################################################################
# WRITE OUT THE FULL UN-NORMALISED VERSION WITH ALL TARGETS AND HEADERS
# ###########################################################################################################
train_df.to_csv('sets/Train_24_hour_full.csv', sep=',', encoding='utf-8', index=False, header=True)
test_df.to_csv('sets/Test_24_hour_full.csv', sep=',', encoding='utf-8', index=False, header=True)


# ###########################################################################################################
#  CREATE A NORMALISATION CONFIGURATION TO 
# ###########################################################################################################

features = train_df.columns.tolist()
unwanted = ["No", "year", "month", "day", "TARGET_pm2.5_24_DIFF", "TARGET_pm2.5_24_PROP_DIFF"]
for x in unwanted :
    features.remove(x)

train_df2 = train_df.loc[:,features]
test_df2 = test_df.loc[:,features]
config = nzr.create_padded_normalization_config(train_df2, 0.05)


# ###########################################################################################################
#
#  GENERATE 3 DIFFERENT TRAINING AND TESTING SETS
#
# ###########################################################################################################

# ###########################################################################################################
#  RAW TARGET NORMALISED 
# ###########################################################################################################

target_col = "TARGET_pm2.5_24_VALUE"
nzr.write_field_config(config, target_col, 'sets/Target_24_nzr_config.yaml')

train_df_norm = nzr.normalize(train_df2, config, ['N','S','E','W'])
test_df_norm = nzr.normalize(test_df2, config, ['N','S','E','W'])

train_df_norm.to_csv('sets/Train_24_hour_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('sets/Test_24_hour_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)


