KDD CUP 98
===================

Downloaded from [https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)


### Source:

Song Xi Chen, csx '@' gsm.pku.edu.cn, Guanghua School of Management, Center for Statistical Science, Peking University.


### Data Set Information:

The data time period is between Jan 1st, 2010 to Dec 31st, 2014. Missing data are denoted as NA

There are a total of 43,824 records.

### Attribute Information:

No: row number 
year: year of data in this row 
month: month of data in this row 
day: day of data in this row 
hour: hour of data in this row 
pm2.5: PM2.5 concentration (ug/m^3) 
DEWP: Dew Point (â„ƒ) 
TEMP: Temperature (â„ƒ) 
PRES: Pressure (hPa) 
cbwd: Combined wind direction 
Iws: Cumulated wind speed (m/s) 
Is: Cumulated hours of snow 
Ir: Cumulated hours of rain 


Relevant Papers:

Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. Proceedings of the Royal Society A, 471, 20150257.



Citation Request:

Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. Proceedings of the Royal Society A, 471, 20150257.


### Processing

The file [process.py](process.py) will process the raw data into a predictive format suitable for treating the problem as a regression task
in which we attempt to predict the value of pm2.5 24 hours in advance.

In addition we convert the field 'cbwd' into 4 indicator variables, one for each of the wind direction componenets (N,S,E,W)

We remove any rows for which the target is NULL, and we use mean value imputation for missing values in the features.

We split out the first 30,000 of the remaining rows for training and test on the remaining data.

We normalise all feature columns by subtracting the mean and dividing by the standard deviation of the training data
(mean and std of training is applied to normalise the test data)


