from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import pickle
import pandas as pd
import time
import random


import sys
sys.path.insert(0, '..')

import gradient_boosting
import collect_data
import profit_curve


app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True
)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    benefit = int(request.form['benefit'])
    cost = int(request.form['cost'])
    text_file = open("benefit.txt", "w")
    text_file.write("{}".format(benefit))
    text_file = open("cost.txt", "w")
    text_file.write("{}".format(cost))
    text_file.close()

    profit_curve.plot_model_profits(benefit, cost)
    # time.sleep(5)
    # pic = 'static/img/temp_profit.png'
    pic = 'static/img/' + str(benefit) + '_' + str(cost) + '.png'
    return render_template('dashboard.html', data=pic)

@app.route('/personalize')
def personalize():
    return render_template('personalize.html')

@app.route('/score', methods=['GET', 'POST'])
def score():
    thresh = int(request.form['thresh'])

    x = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    # request_json = json.loads(x)
    # with open('../data/request.json', 'w') as outfile:
    #     json.dump(request_json, outfile)

    # X = collect_data.clean_data(collect_data.get_data('../data/request.json'))

    # prob = predict_proba(model, X, y)
    prob = random.randint(1,100)
    with open('benefit.txt') as f:
        benefit = int(f.read())
    with open('cost.txt') as f:
        cost = int(f.read())

    expected_pl = prob/100 * (benefit - cost) - (1 - prob/100) * cost
    data_list = [thresh, prob, expected_pl]


    return render_template('score.html', data=data_list)

@app.route('/log', methods=['GET', 'POST'])
def log():
    fraud_val = int(request.form['text_hidden'])
    return render_template('log.html', data=fraud_val)

@app.route('/write', methods=['POST'])
def write():
    # df = pd.read_json('../data/subset.json')
    # df.to_pickle('../data/test_write.pkl')
    return dashboard()




if __name__ == '__main__':

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.connect(("8.8.8.8", 80))
    # ip_address= (s.getsockname()[0])
    # s.close()
    # print("attempting to register {}:{}".format(ip_address, PORT))
    # register_for_ping(ip_address, str(PORT))
    # Start Flask app
    with open('../data/model.pkl', 'rb') as f:
        model = pickle.load(f)


    app.run(host='0.0.0.0', port=8150, debug=True)
