# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:36:28 2013

@author: Edwin
"""

import numpy as np
import matplotlib.pyplot as plt

def smoothed(a):
    b = np.zeros((N/binSize, 1))
    j = 0
    for i in range(0, N, binSize):
        b[j,0] = np.mean(a[i:i+binSize,0])
        j += 1
    return b

N = 18000
binSize = 300

numSamples = 600
nTrials = 18000
initialEpsilon = 1.0
finalEpsilon = 0.1
bDecay = initialEpsilon - 1.0
epsilonDecay = np.exp(np.log(finalEpsilon-bDecay)/(nTrials*numSamples))

colors = ('r','b','g','c','m','k','y','r--','b--','g--','c--','m--','k--','y--','r-.','b-.')

numTest = 15

##########################
### R mean plot
##########################
plt.subplot(1,4,1)
plt.title('Mean reinforcement')
plt.xlabel('Trial')
for FigNum in range(1,numTest+1):
    b = smoothed(np.load('Avs3_R_mean_0%1s.npy'%(FigNum)))
    plt.plot(np.arange(1,N/binSize+1)*binSize, b,colors[FigNum-1])
plt.grid()

##########################
### Total Number of games plot
##########################
plt.subplot(1,4,2)
plt.title('Number of games per trial')
plt.xlabel('Trial')
for FigNum in range(1,numTest+1):
    b = smoothed(np.load('Avs3_gamesT_0%1s.npy'%(FigNum)))
    plt.plot(np.arange(1,N/binSize+1)*binSize, b,colors[FigNum-1])
plt.grid()

##########################
### Win probability, epsilon decay and phi decay plots
##########################
plt.subplot(1,4,3)
plt.title('Winning probability')
plt.xlabel('Trial')
plt.plot(epsilonDecay**(np.arange(0,N)*numSamples),'k')     ## Epsilon decay
for FigNum in range(1,numTest+1):
    b = smoothed(np.load('Avs3_R_win_0%1s.npy'%(FigNum)))
    plt.plot(np.arange(1,N/binSize+1)*binSize, b,colors[FigNum-1])
plt.grid()

###########################
#### Tranfer rate and policy differences
###########################
#plt.subplot(1,4,4)
#plt.title('Transfer')
#plt.xlabel('Trial')
#for FigNum in range(1,numTest+1):
#    b = smoothed(np.load('Avs2_transfNum_0%1s.npy'%(FigNum)))
#    plt.plot(np.arange(1,N/binSize+1)*binSize, b,'r')#colors[FigNum-1])
#    
#    bb = smoothed(np.load('Avs2_transfEqual_0%1s.npy'%(FigNum)))
#    plt.plot(np.arange(1,N/binSize+1)*binSize, bb,'b')#colors[FigNum-1])
#    
#    bbb = 100*smoothed(np.load('Avs2_transfEqual_0%1s.npy'%(FigNum))/np.load('Avs2_transfNum_0%1s.npy'%(FigNum)))
#    plt.plot(np.arange(1,N/binSize+1)*binSize, bbb,'k')#colors[FigNum-1])
#
#plt.grid()


plt.show()
