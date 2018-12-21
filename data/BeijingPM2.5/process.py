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

# NOW REMOVE THE STUFF WE DON'T NEED AND SPLIT OUT TRAIN AND TEST
# final_df.isnull().sum()
nonull_df = final_df[np.isfinite(final_df['pm2.5_target'])]

trainset = 30000
train_df = nonull_df.loc[0:trainset,['day','hour','DEWP','TEMP','PRES','Iws','Is','Ir','N','S','E','W','pm2.5','pm2.5_target']]
test_df = nonull_df.loc[trainset+1:,['day','hour','DEWP','TEMP','PRES','Iws','Is','Ir','N','S','E','W','pm2.5','pm2.5_target']]

# WRITE OUT THE UN-NORMALISED VERSION

train_df.to_csv('train_set.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df.to_csv('test_set.csv', sep=' ', encoding='utf-8', index=False, header=False)


# NOW NORMALISE AND WRITE AGAIN

std_vector = train_df.std()
mean_vector = train_df.mean()

def normalise(indf, means, stds, keepers):
   rez = (indf - means)/stds
   rez = rez.fillna(0)
   for index in keepers:
      rez[index] = indf[index]
   return rez

train_df_norm = normalise(train_df, mean_vector, std_vector, ['N','S','E','W','pm2.5','pm2.5_target'])
test_df_norm = normalise(test_df, mean_vector, std_vector, ['N','S','E','W','pm2.5','pm2.5_target'])

train_df_norm.to_csv('train_set_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)
test_df_norm.to_csv('test_set_norm.csv', sep=' ', encoding='utf-8', index=False, header=False)


# Alternative Normalisation Approach
def minMaxnormalise(indf, cols_to_norm):
   rez = indf.copy()
   rez[cols_to_norm] = rez[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



