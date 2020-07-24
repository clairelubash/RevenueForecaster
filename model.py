
import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib as joblib

from data_ingestion import fetch_ts
from feature_engineering import engineer_features
from logger import update_predict_log, update_train_log

model_dir = os.path.join('models')
model_vers = 0.1


def _model_train(df, tag, test = False):
    
    '''
    loop through and train different models
    '''
    
    time_start = time.time()
    
    X, y, dates = engineer_features(df)
    
    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples,
                                          replace = False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]
        
    X, y, dates = engineer_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)
    
    reg_names = ['RF', 'ADA', 'GB']
    regressors = (RandomForestRegressor(random_state = 100), AdaBoostRegressor(random_state = 100), 
                  GradientBoostingRegressor(random_state = 100))

    params = [
        {'reg__n_estimators': [10, 15, 20, 25],
         'reg__max_features': [3, 4, 5]
        },
        {'reg__n_estimators': [10, 15, 20, 25],
         'reg__learning_rate': [1, 0.1, 0.01, 0.001]
        },
        {'reg__n_estimators': [10, 15, 20, 25],
         'reg__max_features': [3, 4, 5]
        }
    ]

    models = {}
    
    for iteration, (name, regressor, param) in enumerate(zip(reg_names, regressors, params)):
        
        pipeline = Pipeline(steps = [
            ('scaler', StandardScaler()),
            ('reg', regressor)
        ])
        
        grid = GridSearchCV(pipeline, param_grid = param, scoring = 'neg_mean_squared_error', cv = 5, 
                           n_jobs = -1, return_train_score = True)
        
        grid.fit(X_train, y_train)
        
        models[name] = grid, grid.best_estimator_.get_params()
        
    test_scores = []
    
    for key, model in models.items():
        
        y_pred = model[0].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        test_scores.append(rmse)
        
    best_model = reg_names[np.argmin(test_scores)]
    opt_model, params = models[best_model]
    
    country_name = tag.replace('_', ' ').title()
    
    print('Model Results for {}: \n'.format(country_name))
    
    print('RMSE Values:')
    
    for i in range(len(reg_names)):
        print('{}: {} \n'.format(reg_names[i], test_scores[i]))

    print('Best Model: \n {}'.format(next(iter(models.items()))[1][1]['reg']))
    
    print('============================================================')
    
    if test:
        saved_model = os.path.join(model_dir, 'test-{}-model-{}.joblib'.format(tag, re.sub('\.', '_', str(model_vers))))
    else:
        saved_model = os.path.join(model_dir, 'prod-{}-model-{}.joblib'.format(tag, re.sub('\.', '_', str(model_vers))))
                                   
    joblib.dump(opt_model, saved_model)
    
    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = '%03d:%02d:%02d'%(h, m, s)

    update_train_log((str(dates[0]), str(dates[-1])), {'rmse': max(test_scores)}, runtime, model_vers, test = True)
    
    
def model_train(data_dir, test = False):

    '''
    train models for each country and select optimal model
    '''
    
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        
    ts_data = fetch_ts(data_dir)

    for country, df in ts_data.items():
        _model_train(df, country, test = test) 
        
        
def model_load(data_dir = None, training = True):

    '''
    load trained model
    '''

    if not data_dir:
        data_dir = os.path.join('.', 'data', 'cs-train')
    
    models = [f for f in os.listdir(os.path.join('.', 'models')) if f.endswith('.joblib')]

    if len(models) == 0:
        raise Exception('model cannot be found')

    all_models = {}
    
    for model in models:
        all_models[re.split('-', model)[1]] = joblib.load(os.path.join('.', 'models', model))

    ts_data = fetch_ts(data_dir)
    
    all_data = {}
    
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training = training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {'X':X, 'y':y, 'dates': dates}
        
    return(all_data, all_models)


def model_predict(country, year, month, day, all_models = None, test = False):
    
    '''
    predict from model given country and date
    '''
    
    time_start = time.time()
    
    if not all_models:
        all_data, all_models = model_load(training = False)
        
    if country not in all_models.keys():
        raise Exception('model for country {} could not be found'.format(country))

    for d in [year, month, day]:
        if re.search('\D', d):
            raise Exception('invalid year, month, or day')

    model = all_models[country]
    data = all_data[country]
    
    target_date = '{}-{}-{}'.format(year, str(month).zfill(2), str(day).zfill(2))
    print(target_date)
    
    if target_date not in data['dates']:
        raise Exception('date {} not in range {}-{}'.format(target_date, data['dates'][0], data['dates'][-1]))
        
    date_indx = np.where(data['dates'] == target_date)[0][0]
    
    query = data['X'].iloc[[date_indx]]
    
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception('dimensions mismatch')
        
    y_pred = model.predict(query)
    y_proba = None
    
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)
            
    print('y_pred: {}, y_proba: {}'.format(y_pred, y_proba))
     
    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = '%03d:%02d:%02d'%(h, m, s)

    update_predict_log(y_pred, y_proba, target_date, runtime, model_vers, test = test)
    
    return({'y_pred': y_pred, 'y_proba': y_proba})
        
      
if __name__ == '__main__':

    data_dir = os.path.join('.', 'data', 'cs-train')
    model_train(data_dir, test = True)

    all_data, all_models = model_load()
    print('Models Loaded: ',', '.join(all_models.keys()))

    country = 'all'
    year = '2018'
    month = '01'
    day = '05'
    
    result = model_predict(country, year, month, day)
    print(result)   
    