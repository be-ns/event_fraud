import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

def get_labels(label_file='../y.pkl'):
    '''
    Takes a string file path of pickled labels and returns labels in a dataframe.
    '''
    return pd.read_pickle(label_file)

def get_trn_data(data_file='data/tim_X.pkl'):
    '''
    Takes a string file path of pickled data and returns the data in a dataframe.
    '''
    return pd.read_pickle(data_file)

def get_predicted_probs(prediction_file='../predicted_p.pkl'):
    '''
    Takes a string file path of pickled predictions and returns them in a dataframe.
    '''
    return pd.read_pickle(prediction_file)

def get_model(pkl_file='model.pkl'):
    '''
    Takes a string file path of a pickled model and returns the model.
    '''
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def get_cost_benefit(cost_of_fraud, cost_of_investigating):
    '''
    Takes two integers: cost of missing fraud, and cost of investigating fraud
    Returns cost-benefit matrix
    '''
    benefit = cost_of_fraud - cost_of_investigating
    cost = -cost_of_investigating
    return np.array([[benefit, cost],[0, 0]])

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(predicted_probs, labels, cost_fraud, cost_investigate):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and predicted probabilities of data points and thier true labels.

    Parameters
    ----------

    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    cost_benefit = get_cost_benefit(cost_fraud, cost_investigate)
    thresholds = np.linspace(.01, 1, 200)
    profits = []
    flag_rates = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        flagged = np.sum(y_predict)
        flag_rates.append(flagged/y_predict.shape[0])
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / flagged
        # print(confusion_matrix)
        # print(threshold_profit)
        # print('----------------')
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds), np.array(flag_rates)

def plot_model_profits(cost_fraud, cost_investigate, save_path='static/img/'):
    """Plotting function to compare profit curves of different models.

    Parameters
    ----------
    save_path: str, file path to save the plot to. If provided plot will be saved and not shown.
    """
    plt.close('all')
    labels = get_labels()
    predicted_probs = get_predicted_probs()
    profits, thresholds, flag_rates = profit_curve(predicted_probs, labels, cost_fraud, cost_investigate)
    # percentages = np.linspace(0, 100, profits.shape[0])
    # plt.plot(thresholds, profits)
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(thresholds, profits*flag_rates*10000)
    plt.title("Profit Curve")
    plt.xlabel("Flag case if fraud probability is greater than...")
    plt.ylabel("Profit per 10,000 events")
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path+str(cost_fraud)+'_'+str(cost_investigate)+'.png')
    return

# if __name__ == '__main__':
    # plt.close('all')
    # y = get_labels()
    # X = get_trn_data()
    # model = get_model()
    # predicted_probs = model.predict_proba(X)[:, 1]
    # pd.DataFrame(predicted_probs).to_pickle('predicted_p.pkl')
    # plot_model_profits(2000, 300)
