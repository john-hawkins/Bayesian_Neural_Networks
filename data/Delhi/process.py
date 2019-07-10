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

MISSING = object()
# HELPER FUNCTION TO FILL IN THE MISSING DATES
def insert_missing_dates(df, date_field, other_col=MISSING):
    maxdate = max(df[date_field])
    mindate = min(df[date_field])
    mydates = pd.date_range(mindate, maxdate)
    if other_col == MISSING:
        temp = mydates.to_frame(index=False)
        temp.columns=[date_field]
        new_df = pd.merge(temp, df,  how='left', left_on=[date_field], right_on = [date_field])
    else:
        newdf = pd.DataFrame()
        vals = df[other_col].unique()
        for i in range(len(vals)):
            for j in range(len(mydates)):
                newdf = newdf.append({other_col: vals[i], date_field: mydates[j]}, ignore_index=True)
        new_df = pd.merge(newdf, df,  how='left', left_on=[date_field, other_col], right_on = [date_field, other_col])
    return new_df



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

# ADD INDICATOR COLUMNS FOR LOCATION
dfj['Residential'] = np.where(dfj['Type of Location']=='Residential, Rural and other Areas',1,0)
dfj['Industrial'] = np.where(dfj['Type of Location']=='Industrial Area',1,0)


keep_cols = ['Date', 'Stn Code', 'Residential', 'Industrial', 'SO2', 'NO2', 'RSPM/PM10']
df = dfj.loc[:,keep_cols]

expanded = insert_missing_dates(df, 'Date', other_col='Stn Code')

new_names = ['Date', 'Stn Code', 'Residential', 'Industrial', 'SO2', 'NO2', 'PM10']
expanded.columns = new_names

expanded.to_csv( 'delhi_pm10_stations_ALL_2011_2015.csv', header=True, index=False )

# EXTRACT OUT THE INDIVIDUAL SERIES IN THE DATA AND SAVE

stations = df['Stn Code'].unique()

for stat in stations:
    temp = expanded[ expanded['Stn Code']==stat ]
    filename = 'delhi_pm10_station_%s_2011_2015.csv' % str(stat)
    temp.to_csv(filename, header=True, index=False)


# FINALLY BUILD ONE DATASET WITH EACH OF THE PM10 STATION READINGS AS FEATURES
base = pd.DataFrame()
for stat in stations:
    if len(base) == 0 :
        temp = expanded[ expanded['Stn Code']==stat ]
        temp2 = temp.loc[:,['Date','PM10']]
        col_name = 'STN_%s_PM10' % str(stat)
        temp2.columns = ['Date', col_name]
        base = temp2
    else :
        temp = expanded[ expanded['Stn Code']==stat ]
        temp2 = temp.loc[:,['Date','PM10']]
        col_name = 'STN_%s_PM10' % str(stat)
        temp2.columns = ['Date', col_name]
        new_df = pd.merge(base, temp2,  how='left', left_on=['Date'], right_on = ['Date'])
        base = new_df    

filename = 'delhi_pm10_all_stations_wide.csv'
base.to_csv(filename, header=True, index=False)
          
