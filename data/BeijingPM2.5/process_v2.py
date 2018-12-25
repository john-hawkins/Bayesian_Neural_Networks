import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')

#
# BUILD TWO DATASETS 
# - PREDICT SAME HOUR NEXT DAY
# _ PREDICT SAME HOUR NEXT WEEK
#

# TRIM OFF 24 HOURS FROM START (ELIMATE THE NAs)
df2 = df[24:] 

df2['N'] = np.where(df2.cbwd.str[0:1]=='N', 1, 0)
df2['S'] = np.where(df2.cbwd.str[0:1]=='S', 1, 0)
df2['E'] = np.where(df2.cbwd.str[1:2]=='E', 1, 0)
df2['W'] = np.where(df2.cbwd.str[1:2]=='W', 1, 0)

df_f = df2[0:len(df2)-24]

df_t = df2[24:].copy()
df_t['No'] = df_t['No']-24
df_t = df_t.rename(columns={'pm2.5': 'pm2.5_target'})
df_targs = df_t.loc[:,['No','pm2.5_target']]

final_df = pd.merge(df_f, df_targs, left_on='No', right_on='No')
final_df['target_24hr_diff'] = final_df['pm2.5_target'] - final_df['pm2.5']

# TODO: NORMALISE BASED ON THE FIRST 20K Rows


# NOW WE JOIN IT AGAINST ITSELF TO GET MULTIPLE LAGGED OBSERVATIONS
# TO BE USED AS FEATURES FOR PREDICTION

df_t = final_df[72:].copy()
df_t['No'] = df_t['No']-72
df_t_final = df_t.loc[:,['No','target_24hr_diff']]

df_p = final_df[48:len(final_df)-24].copy()
df_p['No'] = df_p['No']-48
df_p = df_p.rename(columns={'pm2.5': 'pm2.5_ref'})
df_p = df_p.rename(columns={'target_24hr_diff': 'pm2.5_diff'})

df_l1 = final_df[47:len(final_df)-25].copy()
df_l1['No'] = df_l1['No']-47
df_l1 = df_l1.rename(columns={'pm2.5': 'pm2.5_Lag1'})
df_l1 = df_l1.rename(columns={'target_24hr_diff': 'pm2.5_diff_L1'})
df_l1_final = df_l1.loc[:,['No','pm2.5_Lag1','pm2.5_diff_L1']]

df_l2 = final_df[46:len(final_df)-26].copy()
df_l2['No'] = df_l2['No']-46
df_l2 = df_l2.rename(columns={'pm2.5': 'pm2.5_Lag2'})
df_l2 = df_l2.rename(columns={'target_24hr_diff': 'pm2.5_diff_L2'})
df_l2_final = df_l2.loc[:,['No','pm2.5_Lag2','pm2.5_diff_L2']]

df_l3 = final_df[45:len(final_df)-27].copy()
df_l3['No'] = df_l3['No']-45
df_l3 = df_l3.rename(columns={'pm2.5': 'pm2.5_Lag3'})
df_l3 = df_l3.rename(columns={'target_24hr_diff': 'pm2.5_diff_L3'})
df_l3_final = df_l3.loc[:,['No','pm2.5_Lag3','pm2.5_diff_L3']]

df_l24 = final_df[24:len(final_df)-48].copy()
df_l24['No'] = df_l24['No']-24
df_l24 = df_l24.rename(columns={'pm2.5': 'pm2.5_Lag24'})
df_l24 = df_l24.rename(columns={'target_24hr_diff': 'pm2.5_diff_L24'})
df_l24_final = df_l24.loc[:,['No','pm2.5_Lag24','pm2.5_diff_L24']]

df_l48 = final_df[0:len(final_df)-72].copy()
df_l48['No'] = df_l48['No']-0
df_l48 = df_l48.rename(columns={'pm2.5': 'pm2.5_Lag48'})
df_l48 = df_l48.rename(columns={'target_24hr_diff': 'pm2.5_diff_L48'})
df_l48_final = df_l48.loc[:,['No','pm2.5_Lag48','pm2.5_diff_L48']]

# MERGE IN THE LAGGED COLUMNS

part_1 = pd.merge(df_p, df_l1_final, left_on='No', right_on='No')
part_2 = pd.merge(part_1, df_l2_final, left_on='No', right_on='No')
part_3 = pd.merge(part_2, df_l3_final, left_on='No', right_on='No')
part_4 = pd.merge(part_3, df_l24_final, left_on='No', right_on='No')
part_5 = pd.merge(part_4, df_l48_final, left_on='No', right_on='No')
final_df = pd.merge(part_5, df_t_final, left_on='No', right_on='No')


# NOW REMOVE ALL ROWS WHERE THE TARGET VALUE IS MISSING
# IN EITHER TARGET OR PREVIOUS VALUE
# final_df.isnull().sum()
nonull_df = final_df[np.isfinite(final_df['target_24hr_diff'])]
nonull_df2 = nonull_df[np.isfinite(nonull_df['pm2.5_diff'])]

trainset = 30000
train_df = nonull_df2.loc[0:trainset,['day','hour','DEWP','TEMP','PRES','Iws','Is','Ir','N','S','E','W','pm2.5_ref','pm2.5_Lag1','pm2.5_diff_L1','pm2.5_Lag2','pm2.5_diff_L2','pm2.5_Lag3','pm2.5_diff_L3', 'pm2.5_Lag24', 'pm2.5_diff_L24', 'pm2.5_Lag48', 'pm2.5_diff_L48', 'pm2.5_diff', 'target_24hr_diff']]


test_df = nonull_df2.loc[trainset+1:,['day','hour','DEWP','TEMP','PRES','Iws','Is','Ir','N','S','E','W','pm2.5_ref','pm2.5_Lag1','pm2.5_diff_L1','pm2.5_Lag2','pm2.5_diff_L2','pm2.5_Lag3','pm2.5_diff_L3', 'pm2.5_Lag24', 'pm2.5_diff_L24', 'pm2.5_Lag48', 'pm2.5_diff_L48', 'pm2.5_diff', 'target_24hr_diff']]


# WRITE OUT THE UN-NORMALISED VERSION

train_df.to_csv('train_set_v2.csv', sep=' ', encoding='utf-8', index=False, header=True)
test_df.to_csv('test_set_v2.csv', sep=' ', encoding='utf-8', index=False, header=True)


# NOW NORMALISE AND WRITE AGAIN

std_vector = train_df.std()
mean_vector = train_df.mean()

def normalise(indf, means, stds, keepers):
   rez = (indf - means)/stds
   rez = rez.fillna(0)
   for index in keepers:
      rez[index] = indf[index]
   return rez

train_df_norm = normalise(train_df, mean_vector, std_vector, ['N','S','E','W'])
test_df_norm = normalise(test_df, mean_vector, std_vector, ['N','S','E','W'])

train_df_norm.to_csv('train_set_v2_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('test_set_v2_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)


# Alternative Normalisation Approach
def minMaxnormalise(indf, cols_to_norm):
   rez = indf.copy()
   rez[cols_to_norm] = rez[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


