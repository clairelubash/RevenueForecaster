
from flask import Flask, jsonify, request, send_from_directory
import os
import argparse
import joblib
import socket
import json
import numpy as np
import pandas as pd

from model import *
from logger import *

app = Flask(__name__)


@app.route('/')
def home():
    return 'Home Page'
    
    
@app.route('/train', methods = ['GET', 'POST'])
def train():
    
    if not request.json:
        return jsonify(False)
    
    data_dir = os.path.join('.', 'data', 'cs-train')
    
    model = model_train(data_dir)
    
    return jsonify(True)

    
@app.route('/predict', methods = ['GET','POST'])
def predict():
    
    if not request.json:
        return jsonify(False)
    
    _result = model_predict(country = request.json['country'], year = request.json['year'], 
                            month = request.json['month'], day = request.json['day'])
    
    result = {}
    
    for key, item in _result.items():
        
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
        
    return(jsonify(result))
    

@app.route('/logs/<filename>', methods = ['GET'])
def logs(filename):

    if not re.search('.log', filename):
        return jsonify(False)

    log_dir = os.path.join('.', 'logs')
    
    if not os.path.isdir(log_dir):
        return jsonify(False)

    file_path = os.path.join(log_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify(False)
    
    return send_from_directory(log_dir, filename, as_attachment = True)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--debug', action = 'store_true', help = 'debug flask')
    args = vars(ap.parse_args())

    if args['debug']:
        app.run(debug = True, port = 8080)
    else:
        app.run(host = '0.0.0.0', threaded = True , port = 8080)