
Full dataset here

https://github.com/LabIA-UFBA/SUNT#

List of stops

https://github.com/LabIA-UFBA/SUNT/blob/main/data/raw/GTFS_INTEGRA_SALVADOR2/stops.txt

### Data Dictionary for Additional Data

Note that you cannot use all predictors to predict the bus loads from the same row -- you will have to delay some predictors by the amount in the 'delay' column in order to use them fairly.

| Field Name                | Data Type             | Explanation                                                                                                                                               | Source            | Frequency  | Delay    |
|---------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|------------|----------|
| `Day_Of_Week`             | Categorical (ordinal) | 0 = Monday, 1 = Tuesday...  6 = Sunday                                                                                                                    |                   | Daily      | Nil      |
| `Public_Holiday`          | Categorical (binary)  | 0 = not a public holiday, 1 = public holiday                                                                                                              |                   | Daily      | Nil      |
| `Hours_Since_Midnight`    | Numeric               | Indication of current time (hours)                                                                                                                        |                   | 5-minutely | Nil      |
| `Flu_Like_Symptoms_Count` | Numeric               | Number of people in Salvador with flu-like symptoms that day                                                                                              | OpenDataSUS       | Daily      | One day  |
| `Dengue_Count`            | Numeric               | Number of people in Salvador with a diagnosis of dengue that day                                                                                          | OpenDataSUS       | Daily      | One day  |
| `ibov_open`               | Numeric               | Opening stock value of the Bovespa Index, Brazil's main stock index. Imputed with the most recent opening stock price if none is available for that hour. | Yahoo Finance API | Hourly     | One hour |
| `temperature_2m`          | Numeric               | Temperature 2m above the ground in Salvador (degrees celsius)                                                                                             | OpenMeteo API     | Hourly     | One hour |
| `precipitation`           | Numeric               | Rainfall in Salvador (mm)                                                                                                                                 | OpenMeteo API     | Hourly     | One hour |  
| `cloudcover`              | Numeric               | Percentage cloud cover in Salvador                                                                                                                        | OpenMeteo API     | Hourly     | One hour |     
| `weathercode`             | Categorical (string)  | String descriptor of the general weather in Salvador                                                                                                      | OpenMeteo API     | Hourly     | One hour |
| `windspeed_10m`           | Numeric               | Wind speed at 10m above the ground in Salvador (km/h)                                                                                                     | OpenMeteo API     | Hourly     | One hour |
| `pm2_5`                   | Numeric               | Particulate matter with diameter smaller than 2.5μm in Salvador (μg/m³). 10m above ground.                                                                | OpenMeteo API     | Hourly     | One hour |
| `ozone`                   | Numeric               | Ozone levels in Salvador 10m above ground (μg/m³).                                                                                                        | OpenMeteo API     | Hourly     | One hour |
| `us_aqi`                  | Numeric               | United States Air Quality Index for Salvador, 0-500.                                                                                                      | OpenMeteo API     | Hourly     | One hour |
| `onibus`                  | Numeric               | Relative search volume in Bahia for ônibus (bus)                                                                                                          | Google Trends     | Daily      | One day  |                           
| `metro`                   | Numeric               | Relative search volume in Bahia for metrô (subway)                                                                                                        | Google Trends     | Daily      | One day  | 
| `festival`                | Numeric               | Relative search volume in Bahia for festival                                                                                                              | Google Trends     | Daily      | One day  | 
| `trem`                    | Numeric               | Relative search volume in Bahia for trem (train)                                                                                                          | Google Trends     | Daily      | One day  | 
| `greve`                   | Numeric               | Relative search volume in Bahia for greve (strike/protest)                                                                                                | Google Trends     | Daily      | One day  | 
