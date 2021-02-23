#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd

def fetchdata(filename):
    # read data
    if filename == 'bullionist.csv':
        data = pd.read_csv('./data/bullionist_controversy/bullionist.csv')
    else:
        data = pd.read_excel('./data/'+filename)
    # set time column as index
    data.rename(columns={list(data)[0]: 'Year'},inplace=True)
    data.set_index('Year',inplace=True)
    # generalize column names
    if len(data.columns) == 2:
        data.rename(columns={list(data)[0]: 'Industrial Production',list(data)[1]: 'CPI'},inplace=True)
    else:
        pass
    return data
 
def data_mat(filename):
    data = fetchdata(filename)
    data = data.to_numpy()
    #growth_rates = (data[1:,:] - data[:-1,:])/(data[:-1,:])
    ## OR with log approx.
    growth_rates = np.log(data[1:,:]) - np.log(data[:-1,:])
    return growth_rates.transpose()

def colname(filename):
    data = pd.read_excel('./data/'+filename)
    col_names = list(data)[1:]
    return col_names
