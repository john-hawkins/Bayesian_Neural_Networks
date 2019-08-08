import numpy as np
import pandas as pd
import datetime

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

df2['Date'] = df['year'].astype(str) + "-" + df['month'].astype(str) + "-" + df['day'].astype(str) + " " +  df['hour'].astype(str) + ":00"

df2['Date'] = df2.apply( lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']).strftime("%Y-%m-%d %H:%M:00"), axis=1 )

features = df2.columns.tolist()
unwanted = ["No", "year", "month", "day"]
for x in unwanted :
    features.remove(x)

final = df2.loc[:,features]

final.to_csv('sets/withDate.csv', sep=',', encoding='utf-8', index=False, header=True)

