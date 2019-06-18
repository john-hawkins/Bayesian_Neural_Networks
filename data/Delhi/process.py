import pandas as pd
import numpy as np
import re

# HELPER FUNCTION FOR INCONSISTENT DATA FORMATS                             
def clean_date(date_val):  
    if re.search('\/.*', date_val): 
        month, day, year = re.split("\/", date_val)      
        if len(year) == 4:
            year = year[2:]
        return day + "-" + month + "-" + year
    else: 
        return date_val   


df1 = pd.read_csv('cpcb_dly_aq_delhi-2011.csv')
df2 = pd.read_csv('cpcb_dly_aq_delhi-2012.csv')
df3 = pd.read_csv('cpcb_dly_aq_delhi-2013.csv')
df4 = pd.read_csv('cpcb_dly_aq_delhi-2014.csv')
df5 = pd.read_csv('cpcb_dly_aq_delhi-2015.csv')
 
# DIFFERENT DATE FORMATS IN THE RAW DATA. NEED TO CLEAN FIRST
df1['Date'] = pd.to_datetime(df1['Sampling Date'], format='%d/%m/%Y')
df2['Date'] = pd.to_datetime(df2['Sampling Date'], format='%d/%m/%Y')
df3['tempdate'] = df3['Sampling Date'].apply( clean_date )  
df3['Date'] = pd.to_datetime(df3['tempdate'], format='%d-%m-%y')
df4['Date'] = pd.to_datetime(df4['Sampling Date'], format='%d-%m-%y')
df5['Date'] = pd.to_datetime(df5['Sampling Date'], format='%d-%m-%y')


# NOW JOIN
dfj = df1.append([df2,df3,df4,df5])

keep_cols = ['Date', 'SO2', 'NO2', 'RSPM/PM10']

df = dfj.loc[:,keep_cols]


df.to_csv( 'delhi_pm10_2011_2015.csv', header=True, index=False )
          
