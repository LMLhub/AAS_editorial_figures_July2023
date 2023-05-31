#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:59:31 2023

@author: Ole Peters and Benjamin Skjold
"""
import os

import numpy as np
#import yaml
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def min_func(x, w, p, risk):
    return p * np.log((w + x - risk)/w) + (1-p)*np.log((w + x)/w)

#**********************************
#********* Set parameters *********
np.random.seed(9274)
data_dir = './'
fig_dir='./'
N = 5 #number of agents
N_exp = [0,N] #numbers of expected-wealth optimizers to be simulated
T = 2500 #number of time units
STEPS=N*T #number of time steps
c = .95 #loss fraction
p = .05 #loss probability

#%% ****************************
#******** Simulate *************
for n_exp in N_exp:
    temporal = np.append(np.zeros(n_exp), np.ones(N-n_exp)) #Boolean indicator for time-average optimizer
    w = np.ones((STEPS,N))  #initialize wealth array

    for t in range(STEPS-1):
        w[t+1] = w[t]               #carry everyone's wealth forward
        i=np.random.randint(0, N)   #select random agent to expose to risk
        risk=w[t][i]*c              #set magnitude of risk

#find fees at which agents offer to insure the risk
        min_fee = np.zeros(N)   
        for n in range(N):
            if temporal[n] == 0:    #expected-wealth optizing agents..
                min_fee[n] = p * risk   #...offer insurance at expected loss.
                if w[t][n]+min_fee[n]-risk<0:   #if risk to be insured can bankrupt the agent..
                    min_fee[n] = np.inf         #...don't offer insurance.
                    
            else: #time-average optimizing agents offer insurance at fee which leaves their time-average growth unchanged
#                min_fee[n] = root_scalar(min_func,args=(w[t][n], p, risk), method='newton', bracket=[risk-w[t][n], risk],xtol=2e-300).root
#                min_fee[n] = root_scalar(min_func,args=(w[t][n], p, risk), method='brentq', bracket=[max(0,risk-w[t][n]), risk],xtol=2e-300).root
                min_fee[n] = root_scalar(min_func,args=(w[t][n], p, risk), method='brentq', bracket=[0, risk],xtol=2e-300).root
        min_fee[i] = np.inf     #exclude self-insurance


        min_fee_j = np.min(min_fee) #find minimum fee
        j = np.argmin(min_fee)      #find agent offering minimum fee

        #find highest fee insurance-seeking agent would pay to remove risk
        max_fee_i=w[t][i]*(1-np.exp(p*(np.log(1-c)))) if temporal[i] != 0 else -np.inf
        #note: we set the max fee of expected-wealth optimizers to -inf to make sure they don't buy
        #insurance. By the insurance puzzle, they never will, but limited floating-point precision
        #means they might in a simulation.

        #update wealths
        win = np.random.uniform(0,1)>p  #determine if loss occurs
        if max_fee_i>min_fee_j:   #if a contract is made
            fee=min_fee_j #set fee at lowest offer
            w[t+1][i]= w[t][i] - fee # deterministic wealth change for insured agent
            if win:
                w[t+1][j]=w[t][j] + fee #update wealth for insurance seller if no loss
            else:
                w[t+1][j] = w[t][j] + fee - risk #update wealth for insurance seller if loss
        #if no contract is made, then i plays the gamble, everyone else stays unchanged.
        else:
            if not win:
                w[t+1][i] = w[t][i] -risk #update wealth if loss for agent i

    with open(os.path.join(data_dir,f'wealth_trajectories_N{N}_Ne{n_exp}.npy'), 'wb') as f:
        np.save(f, w)

#%% ****************************
#******** Plot results *********
w1 = np.load(os.path.join(data_dir,'wealth_trajectories_N5_Ne5.npy'))
w2 = np.load(os.path.join(data_dir,'wealth_trajectories_N5_Ne0.npy'))
#            w3 = np.load(os.path.join(data_dir,'wealth_trajectories_model_2_N2_Ne_0.npy'))
#            w4 = np.load(os.path.join(data_dir,'wealth_trajectories_model_2_N10_Ne_0.npy'))
#            w5 = np.load(os.path.join(data_dir,'wealth_trajectories_model_2_N5_Ne_0.npy'))
STEPS, N = np.shape(w1)
T=int(STEPS/N)
#T, N = np.shape(w1)
#            T3, N3 = np.shape(w3)
#            T4, N4 = np.shape(w4)
#            T5, N5 = np.shape(w5)
fig, ax = plt.subplots(1,1)
# Set the text size globally for the x-axis and y-axis
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.semilogy(range(T),np.exp(range(T)*np.log(1-c*p)), color=(1,0,0), linewidth=1.5, linestyle='--',alpha=1,label=r'expected value') #plot expected wealth.
ax.semilogy(np.arange(0,T,1./N),w1, color='orange', linewidth=1, linestyle='-',alpha=1) #plot non-insurers
ax.semilogy(np.arange(0,T,1./N),w2, color='blue', linewidth=1, linestyle='-',alpha=1) #plot insurance buyers
a = ax.plot([], color='blue', linewidth=1, linestyle='-',alpha=1,label='Time average optimizers')
ax.plot([], color='orange', linewidth=1, linestyle='-',alpha=1,label='Expected value optimizers')
ax.semilogy(range(T),np.exp(range(T)*(p*np.log(1-c))), color='green', linewidth=1.5, linestyle='--',alpha=1,label='time-average uninsured') #plot time-average growth
ax.set(xlabel = 'time', ylabel = 'wealth')

plt.xlim(0,T, auto=False)
plt.ylim(10**-170,10**10, auto=False)
x_limits = ax.get_xlim()
y_limits = ax.get_ylim()
fig.gca().set_aspect(.8*(x_limits[1]-x_limits[0])/np.log10(y_limits[1]/y_limits[0]))

ax.legend(loc = 'lower left')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'Fig3.pdf'), format='pdf')
