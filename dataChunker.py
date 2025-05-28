# 1614273
# Jayden Litolff
# Produces _f30avg (average of the next 6 values in the column)
# Rolling window shrinks at the end of each days data period, instead of continuing into the next
# There is a stupid ammount of redundant calculation here, but it works so...


import pandas as pd

df_5min = pd.read_csv('loader_03-05_2024.csv')

df_5min['Unnamed: 0'] = pd.to_datetime(df_5min['Unnamed: 0'])

df_5min=df_5min.rename(columns={"Unnamed: 0":"timestamp"})

def get_custom_period(ts):
    date = ts.normalize()
    if ts.hour < 5:
        date = date - pd.Timedelta(days=1)
    return date + pd.Timedelta(hours=5)

df_5min['period_start'] = df_5min['timestamp'].apply(get_custom_period)

groups = [g for _, g in df_5min.groupby('period_start')]

for group in groups:

    col_list = list(group.columns)
    col_list.remove('timestamp')
    col_list.remove('period_start')
    
    for col in col_list:
        # Copy non-null values from 'col1' to 'col3'
        group[col+'_f30avg'] = 0
        group[col+'_f30avg'] = group[col].rolling(window=1).mean().shift(-1).combine_first(group[col+'_f30avg'])
        group[col+'_f30avg'] = group[col].rolling(window=2).mean().shift(-2).combine_first(group[col+'_f30avg'])
        group[col+'_f30avg'] = group[col].rolling(window=3).mean().shift(-3).combine_first(group[col+'_f30avg'])
        group[col+'_f30avg'] = group[col].rolling(window=4).mean().shift(-4).combine_first(group[col+'_f30avg'])
        group[col+'_f30avg'] = group[col].rolling(window=5).mean().shift(-5).combine_first(group[col+'_f30avg'])
        group[col+'_f30avg'] = group[col].rolling(window=6).mean().shift(-6).combine_first(group[col+'_f30avg'])

df_5min = pd.concat(groups).sort_values('timestamp')

df_5min = df_5min.round(3).reindex(sorted(df_5min.columns), axis=1)

df_5min.set_index('timestamp', inplace=True)

df_5min.to_csv("output.csv")