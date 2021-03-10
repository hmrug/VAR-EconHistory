#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from numpy.linalg import inv
import pandas as pd
from tools import export, datafeed
import matplotlib.pyplot as plt
pd.set_option('display.precision',2)
plt.rcParams.update({
    'font.size': 16,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Loading up data
data_name = 'USA.xls' # <--- Input
data_mat = datafeed.data_mat(data_name)

# =============================================
# VAR

# Model parameters (Inputs)
p = 12 # Order of VAR
const = 1 # Constant (yes/no)
h = 30 # Forecast horizon

n = np.size(data_mat,0) # No. of variables
N = np.size(data_mat,1) # No. of observations

# OLS estimation (Y=A*Z+U)
def VAROLS(data,n,N,p,const):
    Y = data[:,p:]
    Z = np.zeros([n*p,N-p])
    for jj in np.arange(N-p):
        Z[:,jj]= np.reshape(np.fliplr(data[:, jj:jj+p]),
                            n*p,order='F')
    Z = np.concatenate((np.ones((const,N-p)),Z),axis=0)
    
    # A=(Y*Y')*inv(Z*Z')
    A = np.dot(np.dot(Y,Z.T),inv(np.dot(Z,Z.T)))
    
    Sigma = (np.dot(Y,Y.T) - np.dot(A, np.dot(Z,Y.T)))/(N-n*p-const)
    return A, Sigma

# VAR(1) representation: Selection (Xi) and companion matrix (J)
def VAR1(A,n,p,const):
    A = A[:, const:]
    
    eye = np.eye(n*(p-1))
    zeros = np.zeros((n*(p-1),n))
    mat = np.concatenate((eye,zeros),axis=1)
    
    Xi = np.concatenate((A,mat),axis=0)
    
    J = np.concatenate((np.eye(n),
                        np.zeros((n,n*(p-1)))),
                       axis=1)
    return Xi, J

# Blanchard-Quach decomposition
def BlanchardQuach(Xi,J,Sigma):
    # Reduced form long-run multiplier
    B1 = np.dot(np.dot(J,np.linalg.inv(np.identity(n*p)-Xi)),J.T)
    # Structural long-run multiplier
    temp = np.dot(np.dot(B1,Sigma),B1.T)
    # Impact matrix
    C1 = np.linalg.cholesky(temp)
    S = np.dot(np.linalg.inv(B1),C1)
    return S, C1

# Matrices needed
A, Sigma = VAROLS(data_mat,n,N,p,const)
Xi, J = VAR1(A,n,p,const)
S, C1 = BlanchardQuach(Xi,J,Sigma)

# ---Impulse responses---
epsilon = np.eye(n)
C = np.zeros((h+1,n,n))

for jj in np.arange(h+1):
    temp1 = np.dot(J, np.linalg.matrix_power(Xi,jj))
    temp2 = np.dot(temp1,J.T)
    temp3 = np.dot(temp2, S)
    C[jj:,:,:] = np.dot(temp3,epsilon)
Cs = np.cumsum(C,axis=0)

# Plots of impulse responses
horizon = np.arange(h+1) # x-axis
shock=[r'$\mathrm{\epsilon_S}$',r'$\mathrm{\epsilon_D}$']
response=[r'$\mathrm{Y}$',r'$\mathrm{P}$']

Cs1 = 100*Cs[:,0,0]
Cs2 = 100*Cs[:,0,1]
Cs3 = 100*Cs[:,1,0]
Cs4 = 100*Cs[:,1,1]

fig_IR, ax = plt.subplots(2,2,figsize=[16,12])

ax[0,0].plot(horizon,Cs1,lw='4')
ax[0,0].axhline(y=100*C1[0,0],c='k',lw='2',ls='--')
ax[0,1].plot(horizon,Cs2,lw='4')
ax[0,1].axhline(y=100*C1[0,1],c='k',lw='2',ls='--')
ax[1,0].plot(horizon,Cs3,lw='4')
ax[1,0].axhline(y=100*C1[1,0],c='k',lw='2',ls='--')
ax[1,1].plot(horizon,Cs4,lw='4')
ax[1,1].axhline(y=100*C1[1,1],c='k',lw='2',ls='--')

for i in np.arange(2):
    ax[1,i].set_xlabel('Horizon')
    ax[i,0].set_ylabel('Response (\%)')
    for j in np.arange(2):
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].set_title(shock[j] + r' $\rightarrow$ ' + response[i])
ax[1,0].legend(['Response','Long-run'],
               loc='lower left',
               bbox_to_anchor= (0,-0.28),
               ncol=2,borderaxespad=0,frameon=True)
plt.tight_layout()

# ---Forcast Error Variance Decomposition---

theta = np.zeros((h+1,n,n))
theta[0,:,:] = (Cs[0,:,:]**2) / np.tile(np.reshape((np.sum(Cs[0,:,:]**2,axis=1)),[n,1]), [1,n])

for jj in np.arange(h):
    # numerator
    theta_numerator = np.sum(Cs[:jj+2,:,:]**2, axis=0)
    # denominator
    d1 = np.transpose(np.sum(Cs[:jj+2,:,:]**2, axis=2))
    d2 = np.transpose(np.array([np.sum(d1, axis=1)]))
    theta_denominator = np.tile(d2, [1,n])
    # final theta
    theta[jj + 1, :, :] = theta_numerator / theta_denominator

# Long-run
theta_long = (C1**2) / np.tile(np.reshape((np.sum(C1**2, axis=1)), [n,1]),[1,n])

# Plots of FEVD
theta1 = 100*theta[:,0,0]
theta2 = 100*theta[:,0,1]
theta3 = 100*theta[:,1,0]
theta4 = 100*theta[:,1,1]

fig_FEVD, ax = plt.subplots(2,2,figsize=[16,12])

ax[0,0].plot(horizon,theta1,lw='4')
ax[0,0].axhline(y=100*theta_long[0,0],c='k',lw='2',ls='--')
ax[0,1].plot(horizon,theta2,lw='4')
ax[0,1].axhline(y=100*theta_long[0,1],c='k',lw='2',ls='--')
ax[1,0].plot(horizon,theta3,lw='4')
ax[1,0].axhline(y=100*theta_long[1,0],c='k',lw='2',ls='--')
ax[1,1].plot(horizon,theta4,lw='4')
ax[1,1].axhline(y=100*theta_long[1,1],c='k',lw='2',ls='--')

for i in np.arange(2):
    ax[1,i].set_xlabel('Horizon')
    ax[i,0].set_ylabel('Contribution (\%)')
    for j in np.arange(2):
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].set_title(shock[j] + r' $\rightarrow$ ' + response[i])
ax[1,0].legend(['Contribution','Long-run'],
               loc='lower left',
               bbox_to_anchor= (0,-0.28),
               ncol=2,borderaxespad=0,frameon=True)
plt.tight_layout()

# C(1) table
C1_df = pd.DataFrame(C1,index=['Output','Prices'],columns=['Supply','Demand'])*100

# =============================================
# Exports
export.figure(fig_IR,'IR.pdf')
export.figure(fig_FEVD,'FEVD.pdf')
export.table(C1_df,'C1.txt')
