#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.linalg import toeplitz
from tools import export, datafeed
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
pd.set_option('display.precision',2)
plt.rcParams.update({
    'font.size': 16,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Loading up data
data_name = 'USA.xls'
data = datafeed.fetchdata(data_name)
data.columns = ['IP','CPI']

output = data.iloc[:,0]
prices = data.iloc[:,1]

# Stats
# Yearly data
def reorder_columns(columns, first_cols=[], last_cols=[], drop_cols=[]):
    columns = list(set(columns) - set(first_cols))
    columns = list(set(columns) - set(drop_cols))
    columns = list(set(columns) - set(last_cols))
    new_order = first_cols + columns + last_cols
    return new_order

yearly_data = data.iloc[::12,:]
yearly_data.index = yearly_data.index.strftime('%Y')

pd.set_option('display.precision',1)
yearly_data_pct = yearly_data.pct_change()*100
yearly_data_all = pd.concat([yearly_data,yearly_data_pct],axis=1)
yearly_data_all.columns = ['IP','CPI', 'IP (% change)', 'CPI (% change)']
yearly_data_all = yearly_data_all[reorder_columns(yearly_data_all,
    ['IP', 'IP (% change)'],['CPI','CPI (% change)'])]

# Subsamples
data_preGD = data[:'19290501']
data_GD = data['19290601':]

# Describing data with basic stats
data3 = [data[:'19290501'], data['19290601':], data]
period = ['Pre-Great Depression','Great Depression','All Data: 1919-1939']

stats= []
for i in range(3):
    sts = data3[i].describe()
    median = pd.DataFrame(data3[i].median()).T
    median = median.rename(index={0: 'median'})
    sts = pd.concat([sts.iloc[:2],median,sts.iloc[2:]]).T
    sts.insert(0,' ',period[i])
    sts.set_index([' ',sts.index],inplace=True)
    sts = sts.T
    stats.append(sts)
stats = pd.concat([stats[0],stats[1],stats[2]],axis=1)

# HP filter
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

Y_trend, Y_cycle = HPfilter(output,129600)

# Figure 1 - Industrial Production: data, trend, cycle
timeHorizon = np.array(data.index)

fig_IP, ax = plt.subplots(2,1,figsize=(16,12))
ax[0].plot(timeHorizon,output,lw='5')
ax[0].plot(timeHorizon,Y_trend,lw='3')
ax[1].plot(timeHorizon,Y_cycle,lw='5')

ax[1].axhline(y=0,ls='--',c='k')

ax[0].set_title('Output')
ax[1].set_title('Cycle component')
for i in np.arange(2):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].grid()
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    ax[i].xaxis.set_minor_locator(AutoMinorLocator())
    ax[i].axvline('1929-05-01',lw='1',c='r')
ax[0].legend(['Industrial Production','Trend'],
               loc='best',frameon=1,prop={'size': 12})

ax[1].set_yticks(np.arange(-10,10,2.5))
ax[1].axhline(y=-5,ls='--',c='gray')
ax[1].axhline(y=5,ls='--',c='gray')
plt.tight_layout()

# Figure 2 - CPI
fig_CPI, ax = plt.subplots(1,1,figsize=(16,9))
ax.plot(timeHorizon,prices,lw='5')
ax.axhline(y=60,c='k',ls='--')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_major_locator(MultipleLocator(3))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.axvline('1929-05-01',lw='1',c='r')

plt.grid()
plt.tight_layout()

# =============================================
# Exports
export.figure(fig_IP,'ts_IP.pdf')
export.figure(fig_CPI,'ts_CPI.pdf')
export.table(yearly_data_all,'yearly_data.txt')
export.table(stats,'stats.txt')
