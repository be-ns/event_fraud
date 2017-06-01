from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import pickle
import pandas as pd

app = Flask(__name__)
PORT = 8080
REGISTER_URL = "http://10.3.0.79:5000/register"
DATA = []
TIMESTAMP = []


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    pic = 'static/img/temp_profit.png'
    return render_template('dashboard.html', data=pic)

@app.route('/personalize')
def personalize():
    return render_template('personalize.html')

@app.route('/score', methods=['GET', 'POST'])
def score():
    thresh = int(request.form['thresh'])
    prob = 51
    x = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    request_json = json.loads(x)
    with open('../data/test_request.json', 'w') as outfile:
        json.dump(request_json, outfile)

    data_list = [thresh, prob]


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
