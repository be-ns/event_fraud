import pandas as pd
import last_eleven_lines
import parker_get_data
import clean1
from sklearn.preprocessing import StandardScaler

'''
This script returns a dataframe X of features chosen by each team member as the most important from their assigned subset of the original data, and a dataframe y of labels (fraud or not).
'''

def get_data(filename='data/data.json', return_y = False):
    df = pd.read_json(filename)
    y = df.acct_type.apply(lambda x: 1 if 'fraud' in x else 0)
    ben_data = last_eleven_lines.return_top_three(last_eleven_lines.get_data(filename)[-1])
    parker_data = parker_get_data.get_data(df)
    lindsey_data = clean1.return_import_features(df)
    tim_data = df[['num_order', 'num_payouts']]
    X = pd.concat([ben_data, parker_data, tim_data, lindsey_data], axis=1)
    X = pd.get_dummies(X, columns=['user_type'])
    scaler = StandardScaler()
    numeric_vars = ['tickets_left', 'types_tickets', 'num_previous_payouts',
       'sale_duration2', 'sale_duration', 'num_order', 'num_payouts',
       'created-to-start', 'body_length']
    X[numeric_vars] = scaler.fit_transform(X[numeric_vars].values)
    if return_y:
        return X, y
    else:
        return X

if __name__ == '__main__':
    X, y = get_data(return_y = True)
