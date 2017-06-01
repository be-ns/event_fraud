import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from collect_data import get_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

def build_model():
    '''
    Creates a gradient boosting classification model.
    Inputs:
    -------
    None

    Output:
    -------
    A gradient boosting classification model.
    '''
    return GradientBoostingClassifier()

def _fit(model,X,y):
    '''
    Fits model to feature matrix and target array
    Inputs:
    -------
    X: 2D feature array
    y: 1D target array

    Output:
    -------
    A fitted gradient boosting model.
    '''
    return model.fit(X,y)


def _grid_search(model, params, X, y):
    '''
    Finds the best parameters for model
    Inputs:
    -------
    model: gradient boosting model unfit to data
    params: dictionary of gradient boosting parameters (keys) and list of values
    X: 2D feature array
    y: 1D target array

    Output:
    -------
    Prints the parameters and values that produce the best estimator.
    '''
    cv = GridSearchCV(model, params)
    cv.fit(X_train, y_train)
    print(cv.best_estimator_)

def score_model(model, X, y, scoring='f1'):
    '''
    '''
    print('F1 score: ', f1_score(y_test,y_predict))
    print(sorted(list(zip(gb.feature_importances_, data.columns)), reverse=True))

if __name__ == '__main__':
    # initial trial:
    # F1 score:  0.84142394822
    # [(0.27640674907029367, 'tickets_left'), (0.15951504792994187, 'body_length'),
    # (0.11291484028897442, 'num_order'), (0.097331719008195844, 'num_payouts'),
    # (0.092319824537402159, 'sale_duration2'), (0.07843649729851368, 'types_tickets'),
    # (0.07085152837305915, 'created-to-start'), (0.059851362117277068, 'user_type'),
    # (0.052372431376342204, 'sale_duration'), (0.0, 'num_previous_payouts')]

    # after dummyzing(?)
    # F1 Score:  [ 0.86552567  0.84513806  0.83252427]

    data, target = pd.read_pickle('data/clean_data.pkl'), pd.read_pickle('data/target.pkl')
    data = pd.get_dummies(data, columns=['user_type'])
    # data = data.drop('user_type', axis=1)
    X,y = data.values, target.values
    # X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,stratify=y)

    try:
        with open('data/model.pkl', 'rb') as f:
            gb = pickle.load(f)
    except ImportError:
        gb = GradientBoostingClassifier(random_state=3)
        # gb.fit(X_train,y_train)
        # score_model(gb, X_test, y_test)
        gb.fit(X,y)
        with open('data/model.pkl', 'wb') as f:
            pickle.dump(gb, f)

    print('F1 Score: ', cross_val_score(gb, X, y, scoring='f1'))


    # params = {'max depth':[3,None],'max_features':['sqrt','log2',None],\
    #         'min_samples_leaf': [1, 2, 4],'random_state': [1],'learning_rate':\
    #         np.arange(0.1,1,0.1)}
    # grid_search(gb, params, X_train, y_train)
