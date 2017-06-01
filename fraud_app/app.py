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

    x = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content
    request_json = json.loads(x)
    with open('../data/test_request.json', 'w') as outfile:
        json.dump(request_json, outfile)
    

    return render_template('score.html', data=request_json)

@app.route('/write', methods=['POST'])
def write():
    df = pd.read_json('../data/subset.json')
    df.to_pickle('../data/test_write.pkl')
    return dashboard()

@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}

def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


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
    app.run(host='0.0.0.0', port=8080, debug=True)
