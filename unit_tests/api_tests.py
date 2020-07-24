
import sys
import os
import unittest
from ast import literal_eval
import requests
import re
import numpy as np
import pandas as pd
import json


port = 8080

try:
    requests.post('http://localhost:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    

class ApiTest(unittest.TestCase):
    
    @unittest.skipUnless(server_available, 'local server is not running')
    def test_01_train(self):
        
        query = {'data_dir': './data/cs-train'}
        r = requests.post('http://localhost:{}/train'.format(port), json = query)
        
        train_complete = re.sub('\W+', '', r.text)
        self.assertEqual(train_complete, 'true')
    
    @unittest.skipUnless(server_available, 'local server is not running')
    def test_02_predict(self):
    
        query = {'country': 'all', 'year': '2018', 'month': '1', 'day': '5'}
        r = requests.post('http://localhost:{}/predict'.format(port), json = query)
        response = json.loads(r.text)

        self.assertTrue(isinstance(response['y_pred'][0], float))
    
    @unittest.skipUnless(server_available, 'local server is not running')
    def test_03_logs(self):
    
        file_name = 'train-test.log'
        request_json = {'file': 'train-test.log'}
        
        r = requests.get('http://localhost:{}/logs/{}'.format(port, file_name))
        
        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)
            

if __name__ == '__main__':
    unittest.main()