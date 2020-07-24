
import os
import sys
import re
import time
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd


def fetch_data(data_dir):

    '''
    '''

    if not os.path.isdir(data_dir):
        raise Exception('data directory does not exist')
    if not len(os.listdir(data_dir)) > 0:
        raise Exception('data directory contains zero files')

    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search('\.json', f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    for f,vdf in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
             df.rename(columns = {'StreamID':'stream_id'}, inplace = True)
        if 'TimesViewed' in cols:
            df.rename(columns = {'TimesViewed':'times_viewed'}, inplace = True)
        if 'total_price' in cols:
            df.rename(columns = {'total_price':'price'}, inplace = True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception('columns name could not be matched to correct columns')

    df = pd.concat(list(all_months.values()), ignore_index = True)
    
    years, months, days = df['year'].values, df['month'].values, df['day'].values 
    dates = ['{}-{}-{}'.format(years[i], str(months[i]).zfill(2), str(days[i]).zfill(2)) for i in range(df.shape[0])]
    
    df['invoice_date'] = np.array(dates, dtype = 'datetime64[D]')
    df['invoice'] = [re.sub('\D+','',i) for i in df['invoice'].values]
    
    df.sort_values(by = 'invoice_date', inplace = True)
    
    df.reset_index(drop = True, inplace = True)
    
    return(df)


def convert_to_ts(df_orig, country = None):

    '''
    '''

    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Excpetion('country not found')
    
        mask = df_orig['country'] == country
        df = df_orig[mask]
    else:
        df = df_orig
        
    invoice_dates = df['invoice_date'].values
    
    start_month = '{}-{}'.format(df['year'].values[0], str(df['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(df['year'].values[-1], str(df['month'].values[-1]).zfill(2))
    
    df_dates = df['invoice_date'].values.astype('datetime64[D]')
    
    days = np.arange(start_month, stop_month, dtype = 'datetime64[D]')
    purchases = np.array([np.where(df_dates == day)[0].size for day in days])
    invoices = [np.unique(df[df_dates == day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates == day]['stream_id'].values).size for day in days]
    views =  [df[df_dates == day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates == day]['price'].values.sum() for day in days]
    year_month = ['-'.join(re.split('-', str(day))[:2]) for day in days]

    df_time = pd.DataFrame({'date':days,
                            'purchases':purchases,
                            'unique_invoices':invoices,
                            'unique_streams':streams,
                            'total_views':views,
                            'year_month':year_month,
                            'revenue':revenue})
    return(df_time)


def fetch_ts(data_dir, clean = False):

    '''
    '''

    ts_data_dir = os.path.join(data_dir, 'train-ts')
    
    if clean:
        shutil.rmtree(ts_data_dir)
    if not os.path.exists(ts_data_dir):
        os.mkdir(ts_data_dir)
     
    if len(os.listdir(ts_data_dir)) > 0:
        return({re.sub('\.csv', '', cf)[3:]:pd.read_csv(os.path.join(ts_data_dir, cf)) for cf in os.listdir(ts_data_dir)})

    df = fetch_data(data_dir)

    table = pd.pivot_table(df, index = 'country', values = "price", aggfunc = 'sum')
    
    table.columns = ['total_revenue']
    
    table.sort_values(by = 'total_revenue', inplace = True, ascending = False)
    
    top_ten_countries =  np.array(list(table.index))[:10]
    print(top_ten_countries)

    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search('\.json', f)]
    countries = [os.path.join(data_dir, 'ts-' + re.sub('\s+', '_', c.lower()) + '.csv') for c in top_ten_countries]

    dfs = {}
    
    dfs['all'] = convert_to_ts(df)
    
    for country in top_ten_countries:
        country_id = re.sub('\s+', '_', country.lower())
        file_name = os.path.join(data_dir, 'ts-' + country_id + '.csv')
        dfs[country_id] = convert_to_ts(df, country = country)
  
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir, 'ts-' + key + '.csv'), index = False)
        
    return(dfs)

if __name__ == '__main__':

    run_start = time.time()
    
    data_dir = os.path.join('.', 'data', 'cs-train')

    ts_all = fetch_ts(data_dir, clean = False)

    m, s = divmod(time.time() - run_start, 60)
    h, m = divmod(m, 60)
    
    print('run time:', '%d:%02d:%02d'%(h, m, s))

    print('Data Loaded:')
    
    for key, item in ts_all.items():
        print(key, item.shape)