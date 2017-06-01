import pandas as pd
import last_eleven_lines
import parker_get_data
import clean1

'''
This script returns a dataframe X of features chosen by each team member as the most important from their assigned subset of the original data, and a dataframe y of labels (fraud or not).
'''

def get_data(filename='data/data.json'):
    df = pd.read_json(filename)
    y = df.acct_type.apply(lambda x: 1 if 'fraud' in x else 0)
    ben_data = last_eleven_lines.return_top_three(last_eleven_lines.get_data(filename)[-1])
    parker_data = parker_get_data.get_data(df)
    lindsey_data = clean1.return_import_features(df)
    tim_data = df[['num_order', 'num_payouts']]
    X = pd.concat([ben_data, parker_data, tim_data, lindsey_data], axis=1)
    return X, y

if __name__ == '__main__':
    X, y = get_data()
