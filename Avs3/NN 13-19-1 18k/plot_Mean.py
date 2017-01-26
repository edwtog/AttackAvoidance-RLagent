# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:37:51 2013

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

#a = np.load('WT_Avs1_R_mean_01.npy')
N = 18000#np.size(a)
binSize = 300

numSamples = 600
nTrials = 18000
initialEpsilon = 0.111111
finalEpsilon = 0.111111
bDecay = initialEpsilon - 1.0
epsilonDecay = np.exp(np.log(finalEpsilon-bDecay)/(nTrials*numSamples))

initialPhi = 0.1
finalPhi = 0.1
bDecayPhi = initialPhi - 1.0
phiDecay = np.exp(np.log(finalPhi-bDecayPhi)/(nTrials*numSamples))

numTest = 15
testNum = 0
plt.figure(1,figsize=(17, 12))
#########################################################
### R mean plot
#########################################################
plt.subplot(1,2,1)
plt.ylabel('Mean of reinforcement', fontsize=18)
plt.xlabel('Batch', fontsize=18)
R_mean = np.zeros((N/binSize, numTest))
for FigNum in range(1,numTest+1):
    b = smoothed(np.load('Avs3_R_mean_0%1s.npy'%(FigNum)))
    R_mean[:,FigNum-1] = np.reshape(b, (1,N/binSize))
    #plt.plot(np.arange(1,N/binSize+1)*binSize, b,'b')
plt.errorbar(np.arange(1,N/binSize+1)*binSize, R_mean.mean(axis=1),yerr=R_mean.std(axis=1),color='r')
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.grid()

np.save('Total_Avs3_Mean_R_0%1s' %(testNum),R_mean)

##########################################################
### Total Number of games plot
##########################################################
##plt.subplot(1,4,2)
##plt.title('Number of games per trial')
##plt.xlabel('Trial')
##gamesT = np.zeros((N/binSize, numTest))
##for FigNum in range(1,numTest+1):
##    b = smoothed(np.load('Avs2_gamesT_0%1s.npy'%(FigNum)))
##    gamesT[:,FigNum-1] = np.reshape(b, (1,N/binSize))
##    #plt.plot(np.arange(1,N/binSize+1)*binSize, b,'b')
##plt.errorbar(np.arange(1,N/binSize+1)*binSize, gamesT.mean(axis=1),yerr=gamesT.std(axis=1),color='r')
##plt.grid()

##########################################################
### Win probability, epsilon decay and phi decay plots
##########################################################
plt.subplot(1,2,2)
plt.ylabel('Winning probability', fontsize=18)
plt.xlabel('Batch', fontsize=18)

##plt.plot(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi,'b')
##plt.plot((epsilonDecay**(np.arange(0,N)*numSamples)+bDecay)*(1-(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi)),'k')
##plt.plot((1-(epsilonDecay**(np.arange(0,N)*numSamples)+bDecay))*(1-(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi)),'m')

R_win = np.zeros((N/binSize, numTest))
for FigNum in range(1,numTest+1):
    b = smoothed(np.load('Avs3_R_win_0%1s.npy'%(FigNum)))
    R_win[:,FigNum-1] = np.reshape(b, (1,N/binSize))
    #plt.plot(np.arange(1,N/binSize+1)*binSize, b,'b')
plt.errorbar(np.arange(1,N/binSize+1)*binSize, R_win.mean(axis=1),yerr=R_win.std(axis=1),color='r')
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.ylim(-0.0, 0.7)
plt.grid()

np.save('Total_Avs3_Prob_win_0%1s' %(testNum),R_win)

###########################################################
#### Tranfer rate and policy differences
###########################################################
##plt.subplot(1,4,4)
##plt.title('Transfer')
##plt.xlabel('Trial')
##TransferNum_mean = np.zeros((N/binSize, numTest))
##TransferEqual_mean = np.zeros((N/binSize, numTest))
##TransferRatio_mean = np.zeros((N/binSize, numTest))
##for FigNum in range(1,numTest+1):
##    b = smoothed(np.load('Avs2_transfNum_0%1s.npy'%(FigNum)))
##    TransferNum_mean[:,FigNum-1] = np.reshape(b, (1,N/binSize))
##
##    bb = smoothed(np.load('Avs2_transfEqual_0%1s.npy'%(FigNum)))
##    TransferEqual_mean[:,FigNum-1] = np.reshape(bb, (1,N/binSize))
##    
##    bbb = smoothed(np.load('Avs2_transfEqual_0%1s.npy'%(FigNum))/np.load('Avs2_transfNum_0%1s.npy'%(FigNum)))
##    TransferRatio_mean[:,FigNum-1] = np.reshape(bbb, (1,N/binSize))
##    #plt.plot(np.arange(1,N/binSize+1)*binSize, b,'b')
##plt.errorbar(np.arange(1,N/binSize+1)*binSize, TransferNum_mean.mean(axis=1),yerr=TransferNum_mean.std(axis=1),color='r')
##plt.errorbar(np.arange(1,N/binSize+1)*binSize, TransferEqual_mean.mean(axis=1),yerr=TransferEqual_mean.std(axis=1),color='b')
##plt.errorbar(np.arange(1,N/binSize+1)*binSize, 100*TransferRatio_mean.mean(axis=1),yerr=100*TransferRatio_mean.std(axis=1),color='k')
##plt.grid()
plt.savefig('winProb.png', dpi=300, format='png', bbox_inches='tight')
#plt.show()
