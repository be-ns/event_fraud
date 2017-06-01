import pandas as pd
import last_eleven_lines
import parker_get_data
import clean_1
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import requests
import json
import pickle
from datetime import datetime

'''
This script returns a dataframe X of features chosen by each team member as the most important
from their assigned subset of the original data, and a dataframe y of labels (fraud or not).
'''
with open('data/scaler.pkl','rb') as s:
    scaler = pickle.load(s)

def get_data(filename='data/data.json'):
    '''
    Retrieves data from json file and converts it into a pandas dataframe.
    Inputs:
    -------
    filename: the json file name/path

    Output:
    -------
    Pandas DataFrame
    '''
    df = pd.read_json(filename)
    return df

def clean_data(df, return_y=False):
    '''
    Cleans dataframe
    Input:
    ------
    df: pandas DataFrame
    return_y: bool, if True it creates a target array from 'acct_type' column

    Output:
    -------
    Cleaned pandas DataFrame, and if return_y = True it creates a target array from 'acct_type' column
    '''
    if 'acct_type' in df.columns:
        y = df.acct_type.apply(lambda x: 1 if 'fraud' in x else 0)
    ben_data = last_eleven_lines.return_top_three(last_eleven_lines.get_data(df)[-1])
    parker_data = parker_get_data.get_data(df)
    lindsey_data = clean_first.return_import_features(df)
    tim_data = df[['num_order', 'num_payouts']]
    X = pd.concat([ben_data, parker_data, tim_data, lindsey_data], axis=1)
    X = pd.get_dummies(X, columns=['user_type'])
    scaler = StandardScaler()
    numeric_vars = ['tickets_left', 'types_tickets', 'num_previous_payouts',
       'sale_duration2', 'sale_duration', 'num_order', 'num_payouts',
       'created-to-start', 'body_length','venue_name', 'venue_country',
       'venue_state','user_created']
    X[numeric_vars] = scaler.fit_transform(X[numeric_vars].values)
    if return_y==True:
        return X, y
    return X

def clean_new_data(dic):
    '''
    Reads in a dictionary and converts it into a cleaned pandas DataFrame
    Input:
    ------
    dic: a dictionary whos keys equal columns from original data

    Output:
    -------
    a pandas DataFrame
    '''
    new_dict = {}
    keep_cols = ['body_length','event_created','event_start',
        'num_order','num_payouts',
        'sale_duration2', 'sale_duration']
    for key in dic.keys():
        if key in keep_cols:
            new_dict[key] = dic[key]
    ticket_lst = dic['ticket_types']
    new_dict['tickets_left'] = 0
    for dct in ticket_lst:
        new_dict['tickets_left'] += (dct['quantity_total'] - dct['quantity_sold'])
    new_dict['types_tickets'] = len(ticket_lst)
    new_dict['event_created'] = int(datetime.fromtimestamp(dic['event_created']).strftime('%j'))
    new_dict['event_start'] = int(datetime.fromtimestamp(dic['event_start']).strftime('%j'))
    if new_dict['event_created']>new_dict['event_start']:
        new_dict['created-to-start'] = 365-new_dict['event_created']+new_dict['event_start']
    else:
        new_dict['created-to-start'] = new_dict['event_created']-new_dict['event_start']
    user_type = str(dic['user_type'])
    new_user_type = 'user_type_'+user_type
    new_dict[new_user_type]=1
    return pd.DataFrame(new_dict, columns=new_dict.keys(), index=[0])


def get_column_names(df):
    '''
    Retrieves the column names from a pandas DataFrame
    Input:
    -------
    df: a pandas DataFrame

    Output:
    -------
    an array of column names
    '''
    return df.columns

if __name__ == '__main__':
    # in keys:      ['body_length', 'num_order', 'num_payouts', 'sale_duration', 'sale_duration2']
    # not in keys:  ['approx_payout_date', 'channels', 'country', 'currency', 'delivery_method',
            # 'description', 'email_domain', 'event_created', 'event_end', 'event_published', 'event_start',
            # 'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo', 'listed', 'name',
            # 'name_length', 'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',
            # 'payee_name', 'payout_type', 'previous_payouts', 'show_map', 'ticket_types', 'user_age',
            # 'user_created', 'user_type', 'venue_address',   'venue_country', 'venue_latitude',
            # 'venue_longitude', 'venue_name', 'venue_state']

    # x = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    # x_json = json.loads(x)
    # client = MongoClient()
    # db = client.pymongo_test
    # post_to_mongo(x_json, db)

    # cols = ['tickets_left', 'types_tickets', 'num_previous_payouts',
    #    'sale_duration2', 'sale_duration', 'num_order', 'num_payouts',
    #    'created-to-start', 'body_length', 'user_type_1', 'user_type_2',
    #    'user_type_3', 'user_type_4', 'user_type_5', 'user_type_103']
    # df = get_data()
    # X,y = clean_data(df, return_y=True)
    # cols = get_column_names(X)
    # print(X.head())

    # print(clean_new_data(x_json))
    pass
