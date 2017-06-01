import pandas as pd
import numpy as np

def get_data(df):
    df = df.iloc[:,24:33]

    df['org_facebook'].fillna(df['org_facebook'].mean(), inplace=True)
    df['org_twitter'].fillna(df['org_twitter'].mean(), inplace=True)
    df['sale_duration'].fillna(df['sale_duration'].mean(), inplace=True)
    df['num_previous_payouts'] = -1

    for i in df.index.values:
        df.loc[i,'num_previous_payouts'] = len(df.loc[i,'previous_payouts'])
        print(i)

    return df[['num_previous_payouts', 'sale_duration2', 'sale_duration']]
