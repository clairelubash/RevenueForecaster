
import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from data_ingestion import fetch_ts


def engineer_features(df, training = True):

    '''
    convert data into dictionary and adding features for previous days revenue
    '''

    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    eng_features = defaultdict(list)
    previous = [7, 14, 28, 70]
    y = np.zeros(dates.size)
    
    for d, day in enumerate(dates):

        for num in previous:
            
            current = np.datetime64(day, 'D') 
            prev = current - np.timedelta64(num, 'D')
            
            mask = np.in1d(dates, np.arange(prev, current, dtype = 'datetime64[D]'))
            eng_features['previous_{}'.format(num)].append(df[mask]['revenue'].sum())
 
        plus_30 = current + np.timedelta64(30, 'D')
    
        mask = np.in1d(dates, np.arange(current, plus_30, dtype = 'datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        start_date = current - np.timedelta64(365, 'D')
        stop_date = plus_30 - np.timedelta64(365, 'D')
        
        mask = np.in1d(dates, np.arange(start_date, stop_date, dtype = 'datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        minus_30 = current - np.timedelta64(30, 'D')
        
        mask = np.in1d(dates, np.arange(minus_30, current,dtype = 'datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    X.fillna(0, inplace = True)
    
    mask = X.sum(axis = 1) > 0
    
    X = X[mask]
    y = y[mask]
    
    dates = dates[mask]
    
    X.reset_index(drop = True, inplace = True)

    if training == True:
        
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop = True, inplace = True)
    
    return(X, y, dates)


if __name__ == '__main__':
    
    run_start = time.time()
    
    data_dir = os.path.join('.', 'data', 'cs-train')
    df = fetch_ts(data_dir)
    
    m, s = divmod(time.time() - run_start, 60)
    h, m = divmod(m, 60)
    
    print('run time:', '%d:%02d:%02d'%(h, m, s))
    print('feature engineering complete')