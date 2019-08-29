Delhi Air Quality TIme Series
-----------------------------

This data was was retrieved manually in June 2019 from the Indian Government website:

https://data.gov.in/catalog/historical-daily-ambient-air-quality-data

The API was not used because it appears to require an Indian phone number to register.
Without registration you are restricted to 10 rows per day using a test key.


Preparation
-----------

There are numerous problems with the raw data, different date formats, large missing sections
for certain mnonitoring stattions and irregularity in the time intervals.

The first stage of preparation happens in the file [process.py](process.py)

