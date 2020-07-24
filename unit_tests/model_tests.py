
import unittest, random

from model import *


class ModelTest(unittest.TestCase):
    
    def test_01_train(self):
        
        data_dir = './data/cs-train'
        model_dir = './models'
        
        model_train(data_dir)
        models = [f for f in os.listdir(model_dir) if re.search('test', f)]
        
        self.assertEqual(len(models), 11)
        
    def test_02_load(self):    
        
        all_data, all_models = model_load()
        models_loaded = list(all_models.keys())
        
        model = all_models[random.choice(models_loaded)]
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))
        
    def test_03_predict(self):  
        
        country = 'all'
        year = '2018'
        month = '01'
        day = '05'
        
        result = model_predict(country, year, month, day)
        y_pred = result['y_pred']
        
        self.assertTrue(y_pred.dtype == np.float64)
        
        
if __name__ == '__main__':
    unittest.main()