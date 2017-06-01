import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as rfc
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc


def get_data(df):
    '''
    Reads in dats from the filename (as a JSON file) and then passes a shortened Pandas Dataframe to `parse_ticket_types`.
    INPUT: filepath
    OUTPUT: Passes shortened dataframe to `parse_ticket_types`
    '''
    # reduce DF to only last eleven columns
    df2 = df[['show_map', 'ticket_types', 'user_age', 'user_created', 'user_type','venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'fraud']]
    df2['venue_address'] = df2.venue_address.apply(lambda x: 1 if x != '' else 0)
    df2['same_country'] = df.country == df.venue_country
    df2['normal_age'] = df2.user_age.apply(lambda age: 1 if age <= 75 and age >= 15 else 0)
    df2['venue_geo_stats'] = df2.venue_latitude.apply(lambda lat: 0 if lat == 0.0 else 1)
    df2.fraud = df2.fraud.astype(bool)
    df2.drop(['user_created'], axis=1, inplace=True)
    df2.show_map = df2.show_map.astype(bool)
    df2.venue_address = df2.venue_address.astype(bool)
    df2.drop(['venue_name', 'venue_country', 'venue_state'], axis=1, inplace=True)
    df2.venue_geo_stats = df2.venue_geo_stats.astype(bool)
    df2.normal_age = df2.normal_age.astype(bool)
    return _parse_ticket_types(df2)

def _parse_ticket_types(df2):
    new_column = []
    len_list = []

    for i, row in enumerate(df2.ticket_types.as_matrix()):
        new_row_val = {'availability': 0,'quantity_sold': 0}
        for dct in row:
            new_row_val['availability'] += dct['availability']
            new_row_val['quantity_sold'] += dct['quantity_sold']
        new_column.append(new_row_val['availability'] - new_row_val['quantity_sold'])
        len_list.append(len(row))

    df2['tickets_left'] = np.array(new_column).T
    df2['types_tickets'] = np.array(len_list).T

    df2.drop(['show_map', 'user_age', 'venue_latitude', 'venue_longitude'], axis=1, inplace=True)

    X = df2[['user_type','venue_address','same_country','normal_age','venue_geo_stats','tickets_left', \
            'types_tickets']].as_matrix()
    y = df2.fraud.as_matrix()
    return _split(X, y, df2)

def _split(X, y, df2):
    x_tr, x_te, y_tr, y_te = tts(X, y)
    # x = s_s()
    # x.fit(x_tr)
    # x_tr = x.transform(x_tr)
    # x_te = x.transform(x_te)
    return x_tr, x_te, y_tr, y_te, df2

def build_model(x_tr, y_tr):
    r_fc = rfc(loss='deviance', learning_rate=0.1, n_estimators=500, subsample=1.0, \
           criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, \
           min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_split=1e-07, \
           init=None, random_state=None, max_features=4, verbose=1, max_leaf_nodes=None, \
           warm_start=False, presort='auto')
    r_fc.fit(x_tr, y_tr)
    return r_fc

def score_model(model, x_te, y_te):
    predicted = model.predict(x_te)

    f_score = f1(y_te, predicted)
    accuracy = acc(y_te, predicted)

    # print(sorted(list(zip(model.feature_importances_, ['user_type','venue_address','same_country', \
    # 'normal_age','venue_geo_stats', 'tickets_left','types_tickets'])), reverse=True))
    return f_score, accuracy

def return_top_three(df2):
    return df2[['tickets_left', 'types_tickets', 'user_type']]

if __name__ == '__main__':
    filename = '../fraud-detection-case-study/data.json'
    df = pd.read_json(filename)
    df2 = get_data(df)[-1]
    df3 = return_top_three(df2)
    # model = build_model(x_tr, y_tr)
    # f_score, accuracy = score_model(model, x_te, y_te)
    # print('\n')
    # print('f_score = ', f_score)
    # print('\n')
    # print('accuracy = ', accuracy)
