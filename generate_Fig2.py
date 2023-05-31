# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Ole Peters
"""
import numpy as np
import matplotlib.pyplot as plt

N=100   #number of realizations
T=1000  #number of rounds
np.random.seed(1934)

#compute deterministic trajectories
play_round=list(range(T))
x=np.ones([T,N])
expectation=np.power(1.05,play_round)
t_factor=np.sqrt(1.5*0.6)
time_ave=np.power(t_factor,play_round)

#compute stochastic trajectories
for n in range(0,N):
    for t in range(1,T):
        x[t,n]=x[t-1,n]*np.random.choice([0.6,1.5])

#plot figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(x,color='orange',alpha=.4,linewidth=0.3)           #stochastic trajectories
plt.plot([0, 1000],[1, 1], color='#555555',linewidth=1)     #plot x axis
plt.plot([250, 250],[1, 40], color='#555555',linewidth=1)   #plot x ticks
plt.plot([500, 500],[1, 40], color='#555555',linewidth=1)
plt.plot([750, 750],[1, 40], color='#555555',linewidth=1)
plt.plot([1000, 1000],[1, 40], color='#555555',linewidth=1)
ax.annotate('$250$', xy=(220,50), xytext=(220, 100),fontsize=12)        #label x ticks
ax.annotate('$500$', xy=(470,50), xytext=(470, 100),fontsize=12)
ax.annotate('$750$', xy=(720,50), xytext=(720, 100),fontsize=12)
ax.annotate('$1,000$', xy=(950,50), xytext=(950, 100),fontsize=12)
ax.annotate(r'time', xy=(950,50), xytext=(600, 10**6),fontsize=12)      #label x axis
ax.annotate(r'wealth',rotation=90, xy=(0,50), xytext=(-150, 10**-5),fontsize=12)    #label y axis

plt.plot(expectation,color='r',label=r'expected value',linewidth=2,linestyle='--') #expected value
plt.plot(time_ave,color='g',label=r'time average growth',linewidth=2,linestyle='--') #time-average growth
plt.ylim([np.power(10.,-40),np.power(10.,24)])
plt.yscale('log')

# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('zero')
ax.legend(loc='upper center',bbox_to_anchor=(0.27,1.03))

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')

yticklocs=list([np.double(10**-30),np.double(10**-20),np.double(10**-10),np.double(1),np.double(10**10),np.double(10**20)])
# Show ticks in the left and lower axes only
ax.yaxis.set_ticks_position('left')
plt.yticks(yticklocs,[r'$10^{-30}$',r'$10^{-20}$',r'$10^{-20}$',r'$1$',r'$10^{10}$',r'$10^{20}$'])
ax.tick_params(axis='y', labelsize=12)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.savefig("./Fig2.pdf", bbox_inches='tight')


