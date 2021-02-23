#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from numpy.linalg import inv
from scipy.linalg import toeplitz
import pandas as pd
from tools import export, datafeed
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.precision',2)
sns.set_theme()
plt.rcParams.update({'font.size': 16}) 

data_name = 'USA.xls'
data = datafeed.fetchdata(data_name)

data_yearly = data.iloc[::12,:]
data_stats = data.describe()

def HPfilter(data, mu):
    data = np.matrix(data.to_numpy()).T
    N = len(data)
    a = mu
    b = -4*mu
    c = 1+6*mu
    f = [[c,b,a]]
    z = [np.zeros(N-3)]
    d = np.concatenate((f,z),axis=1)
    A = toeplitz(d)
    A[:2,:2] = [[1+mu,-2*mu],[-2*mu,1+5*mu]]
    A[-2:,-2:] = np.rot90(A[:2,:2],2)
    trend = inv(A)*data
    cycle = data - trend
    return trend, cycle

output = data.iloc[:,0]
prices = data.iloc[:,1]
Y_trend, Y_cycle = HPfilter(output,1600)
timeHorizon = np.array(data.index)

fig_ip, ax = plt.subplots(2,1,figsize=(16,9))
ax[0].plot(timeHorizon,output,lw='3')
ax[0].plot(timeHorizon,Y_trend,lw='2')
ax[1].plot(timeHorizon,Y_cycle,lw='4')
ax[1].axhline(y=0,lw='2',ls='--',c='k')
plt.tight_layout()

fig_cpi, ax = plt.subplots(1,1,figsize=(16,9))
ax.plot(timeHorizon,prices,lw='4')
ax.axhline(y=60,c='k',ls='--')
plt.tight_layout()

# =============================================
# Exports
export.figure(fig_ip,'IP.pdf')
export.figure(fig_cpi,'CPI.pdf')
export.table(data_yearly,'data_yearly.txt')
export.table(data_stats,'data_stats')
