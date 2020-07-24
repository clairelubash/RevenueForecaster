
import re
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_ingestion import fetch_ts


def create_plots(df):
    
    plt.style.use('ggplot')
    
    # total monthly revenue 
    df['all'].groupby(['year_month']).sum()['revenue'].plot()
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.title('Monthly Revenue for All Markets')
    plt.show()
    
    # revenue breakdown by country
    x = list(df.keys())
    x.remove('all')
    x_pos = [i for i, _ in enumerate(x)]
    
    rev_vals = []
    for country in x:
        rev_vals.append(df[country]['revenue'].sum())

    plt.bar(x_pos, rev_vals, color = 'lightblue')
    plt.xlabel('Country')
    plt.ylabel('Revenue')
    plt.yscale('log')
    plt.title('Revenue Breakdown by Country')
    plt.xticks(x_pos, x, rotation = 70)
    plt.show()
    
    # heatmap
    corr = (df['all']).corr()
    
    sns.heatmap(corr, annot = True, cmap = 'Blues')
    plt.show()

if __name__ == '__main__':
    
    run_start = time.time()
    
    data_dir = os.path.join('.', 'data', 'cs-train')
    df = fetch_ts(data_dir)
    
    create_plots(df)
    
    m, s = divmod(time.time() - run_start, 60)
    h, m = divmod(m, 60)
    
    print('run time:', '%d:%02d:%02d'%(h, m, s))
    print('data visualization complete')