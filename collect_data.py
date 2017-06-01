import pandas as pd
import last_eleven_lines
import parker_get_data
import clean_first
from sklearn.preprocessing import StandardScaler

'''
This script returns a dataframe X of features chosen by each team member as the most important 
from their assigned subset of the original data, and a dataframe y of labels (fraud or not).
'''

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
       'created-to-start', 'body_length']
    X[numeric_vars] = scaler.fit_transform(X[numeric_vars].values)
    if return_y==True:
        return X, y
    return X

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
    df = get_data()
    X,y = clean_data(df, return_y=True)
    cols = get_column_names(X)
    print(X.head())
