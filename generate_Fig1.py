#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:54:25 2018

@author: Ole Peters
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T=100
N=5
m=1.
np.random.seed(1934)

noise=pd.DataFrame(np.random.normal(0,1,size=[N,T+1]),index=range(1,N+1))
x=pd.DataFrame(index=range(1,N+1),columns=range(0,T+1))
x.iloc[:,0]=np.random.normal(0,1,size=N)

for t in range(1,T+1):
    x.iloc[:,t]=x.iloc[:,t-1]*(1-m)+noise.iloc[:,t-1]

fig=plt.figure()

ax=x.iloc[1,:].plot(linestyle='-',color='black')
plt.plot(range(0,T+1),x.iloc[2,:]-10,linestyle='-',color='black')
plt.plot(range(0,T+1),x.iloc[3,:]-20,linestyle='-',color='black')
plt.plot(range(0,T+1),x.iloc[4,:]-30,linestyle='-',color='black')
plt.plot(range(0,T+1),x.iloc[0,:]-40,linestyle='-',color='black')

plt.ylim([-55,20])
plt.xlim([-15,111])
plt.axis('off')

ax.annotate('',\
            xy=(105,15), xycoords='data',\
            xytext=(-300, 0), textcoords='offset points',\
            arrowprops=dict(color='green',arrowstyle="->",linewidth=2))
ax.annotate(r' $\hspace{17.4}$ ',\
            xy=(110,-21), xycoords='data',\
            xytext=(-290, 0), textcoords='offset points',\
            size=15,\
            bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.3))
ax.annotate('',\
            xy=(108,-19.5), xycoords='data',\
            xytext=(-19, 0), textcoords='offset points',\
            arrowprops=dict(color='green',arrowstyle="->",linewidth=2))
ax.annotate('',\
            xy=(-15,-50), xycoords='data',\
            xytext=(0, 160), textcoords='offset points',\
            arrowprops=dict(color='red',arrowstyle="->",linewidth=2))
ax.annotate('',\
            xy=(45,-52.5), xycoords='data',\
            xytext=(0, 20), textcoords='offset points',\
            arrowprops=dict(color='red',arrowstyle="->",linewidth=2,linestyle='-'))
ax.annotate(r' $\hspace{39}$ ',\
            xy=(45,-55), xycoords='data',\
            xytext=(-2, 28), textcoords='offset points',\
            size=4,\
            bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3),rotation=90)
ax.annotate('time',\
            xy=(93,3), xycoords='data',\
            xytext=(-20, 15), textcoords='offset points',size=15)
ax.annotate('realization',\
            xy=(-15,-50), xycoords='data',\
            xytext=(0, 20), textcoords='offset points',rotation=-90,size=15)
plt.plot(range(0,T+1),x.iloc[1,:],linestyle='-',color='orange')
plt.plot(range(0,T+1),x.iloc[2,:]-10,linestyle='-',color='orange')
plt.plot(range(0,T+1),x.iloc[3,:]-20,linestyle='-',color='orange')
plt.plot(range(0,T+1),x.iloc[4,:]-30,linestyle='-',color='orange')
plt.plot(range(0,T+1),x.iloc[0,:]-40,linestyle='-',color='orange')
plt.xlabel('r$t$',fontsize=16)
plt.savefig("./Fig1.pdf", bbox_inches='tight')
