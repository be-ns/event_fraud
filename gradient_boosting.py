import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from collect_data import get_data, clean_data, get_column_names
# from sklearn.model_selection import GridSearchCV, cross_val_score

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
	X: 2D data/feature array to fit to model
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
	model: gradient boosting model to fit the data
	params: dictionary of gradient boosting parameters (keys) and list of values
	X: 2D data/feature array to fit to model
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
	Get the 3-fold cross validation score of model
	Inputs:
	-------
	model: gradient boosting model to fit the data
	X: 2D data/feature array to fit to model
	y: 1D target array
	scoring: a string or scoring metric; e.g. 'f1', 'accuracy', 'precision', 'recall', etc.

	Outputs:
	--------
	The model's score
	'''
	print('{0} score: {1}'.format(scoring, cross_val_score(model,X,y,scoring=scoring)))

def feature_importance(model, X, y, column_names):
	'''
	Returns a list of the important features sorted from greatest to least relevance
	Inputs:
	-------
	model: gradient boosting model to fit the data
	X: 2D data/feature array to fit to model
	y: 1D target array
	column_names: array or list of data column names

	Output:
	-------
	List of important features
	'''
	model = _fit(model,X,y)
	print(sorted(list(zip(model.feature_importances_, data.columns)), reverse=True))

def _predict_proba(model, X, y):
	'''
	Predicts the probability that each point data is fraudulent
	Inputs:
	-------
	model: gradient boosting model to fit to data
	X: 2D data/feature array to fit to model
	y: 1D target array

	Output:
	-------
	probability of fraud
	'''
	model = _fit(model,X,y)
	return model.predict_proba(X)[1]

def predict_new_proba(model, X_pred):
	'''
	Predicts the probability that new data (X_pred) is fraudulent
	Inputs:
	-------
	model: gradient boosting model fitted to data
	X_pred: pandas DataFrame or Series to predict on

	Output:
	-------
	probability of fraud
	'''
	X = get_data(X_pred).values
	return model.predict_proba(X)[1]

# if __name__ == '__main__':
	# initial trial:
	# F1 score:  0.84142394822
	# [(0.27640674907029367, 'tickets_left'), (0.15951504792994187, 'body_length'),
	# (0.11291484028897442, 'num_order'), (0.097331719008195844, 'num_payouts'),
	# (0.092319824537402159, 'sale_duration2'), (0.07843649729851368, 'types_tickets'),
	# (0.07085152837305915, 'created-to-start'), (0.059851362117277068, 'user_type'),
	# (0.052372431376342204, 'sale_duration'), (0.0, 'num_previous_payouts')]

	# after dummyzing(?)
	# F1 Score:  [ 0.86552567  0.84513806  0.83252427]

	# df = get_data()
	# data, target = clean_data(df, return_y=True)
	# data.to_pickle('data/clean_data.pkl')
	# target.to_pickle('data/target.pkl')

	# data, target = pd.read_pickle('data/clean_data.pkl'), pd.read_pickle('data/target.pkl')

	# cols = get_column_names(data)
	# X,y = data.values, target.values

	# gb = build_model()
	# gb = _fit(gb, X, y)
	# with open('data/model.pkl', 'wb') as f:
	# 	pickle.dump(gb, f)

	# with open('data/model.pkl', 'rb') as f:
	# 	gb = pickle.load(f)

	# score_model(gb, X, y)

	# params = {'max depth':[3,None],'max_features':['sqrt','log2',None],\
	#         'min_samples_leaf': [1, 2, 4],'random_state': [1]'learning_rate':\
    #         np.arange(0.1,1,0.1)}
    # grid_search(gb, params, X_train, y_train)
