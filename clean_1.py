import pandas as pd
import numpy as np
from collections import Counter
import pickle
from datetime import datetime


def get_counts(df):
    print('account type: ',Counter(data['acct_type']))
    fraud_cnt = [Counter(fraud[i]).most_common(5) for i in ['country','currency','email_domain']]
    not_fraud_cnt = [Counter(not_fraud[i]).most_common(5) for i in ['country','currency','email_domain']]
    print('\nfraud')

# data['event_published'].fillna(0).apply(lambda x: int(x))

def convert_time(df, col):
    date_lst = []
    for i in range(df.shape[0]):
        date_lst.append(int(datetime.fromtimestamp(df.loc[i,col]).strftime('%j')))
    df[col] = np.array(date_lst)
    return df

def time_to_dummy(df, col1, col2, name):
    dummy = []
    for i in range(len(df[col1])):
        if df[col1][i] < df[col2][i]:
            dummy.append(df[col2][i]-df[col1][i])
        else:
            dummy.append(365-df[col1][i]+df[col2][i])
    df[name] = np.array(dummy)
    return df

def clean_data(df):
    df['fraud'] = df.acct_type.apply(lambda x: 1 if 'fraud' in x else 0)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 1 if x==1.0 else 0)
    fraud = df[df['fraud']==1]
    not_fraud = df[df['fraud']==0]

    df1 = df.loc[:,'approx_payout_date':'event_start']
    df2 = pd.concat([df['fraud'],df1], axis=1)

    for date_col in ['approx_payout_date','event_created','event_start']:
        df2 = convert_time(df2, date_col)
    df2 = time_to_dummy(df2, 'event_start', 'approx_payout_date', 'start-to-payout')
    df2 = time_to_dummy(df2, 'event_created', 'event_start', 'created-to-start')
    df2.drop(['country','currency','description','approx_payout_date','event_start',\
        'email_domain','event_published','event_end','event_created'], axis=1, inplace=True)
    return df2

def return_import_features(df):
    df = clean_data(df)
    return df[['created-to-start', 'body_length']]


if __name__ == '__main__':
    data = pd.read_json('data/data.json')

    # my columns = 'approx_payout_date', 'body_length', 'channels', 'country',
    # 'currency', 'delivery_method', 'description', 'email_domain',
    # 'event_created', 'event_end', 'event_published', 'event_start'

    # print(fraud.describe())
    # print(not_fraud.describe())

    # print(fraud_cnt)
    # print(not_fraud_cnt)

    # date_lst = ['event_created', 'event_end', 'event_start']
    # convert_time(data, date_lst)
    # print(data.head())

    df = clean_data(data)
    df1 = return_import_features(df)
    print(df1.head())
