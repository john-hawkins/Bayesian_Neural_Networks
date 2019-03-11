import numpy as np
import pandas as pd
import DatasetGenerator as dg
import Normalizer as nzr

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
list_of_lags = [24,48,72]

new_df = dg.generate_time_series_dataset( df2, index_column, forecast_column, forecast_period, list_of_lags)

trainset = 30000
train_df = new_df.loc[0:trainset,:]
test_df = new_df.loc[trainset+1:,:]

# ###########################################################################################################
# WRITE OUT THE UN-NORMALISED VERSION
# ###########################################################################################################
train_df.to_csv('Train_set_24_hour.csv', sep=' ', encoding='utf-8', index=False, header=True)
test_df.to_csv('Test_set_24_hour.csv', sep=' ', encoding='utf-8', index=False, header=True)

# ###########################################################################################################
# NORMALIZE AND WRITE TO DISK
# ###########################################################################################################
config = nzr.create_normalization_config(train_df)

train_df_norm = nzr.normalize(train_df, config, ['N','S','E','W'])
test_df_norm = nzr.normalize(test_df, config, ['N','S','E','W'])

train_df_norm.to_csv('Train_set_24_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('Test_set_24_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)

