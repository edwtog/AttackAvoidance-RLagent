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
N = 6000#np.size(a)
binSize = 300

numSamples = 600
nTrials = 6000
initialEpsilon = 0.125
finalEpsilon = 0.125
bDecay = initialEpsilon - 1.0
epsilonDecay = np.exp(np.log(finalEpsilon-bDecay)/(nTrials*numSamples))

initialPhi = 0.2
finalPhi = 0.2
bDecayPhi = initialPhi - 1.0
phiDecay = np.exp(np.log(finalPhi-bDecayPhi)/(nTrials*numSamples))

colors = ('r','b','g','c','m','k','y','r--','b--','g--','c--','m--','k--','y--','r-.','b-.')

numTest = 15
iniFig = 0
endFig = 9

#########################################################
### R mean plot
#########################################################
#plt.subplot(1,4,1)
plt.plot(1)
plt.title('Avs3 transfer from Avs2 & Avs0 - NN Avs3 13-19-1')
plt.ylabel('Mean reinforcement')
plt.xlabel('Trial')
#R_mean = np.zeros((N/binSize, numTest))
#
#plt.errorbar(np.arange(1,N/binSize+1)*binSize, R_mean.mean(axis=1),yerr=R_mean.std(axis=1),color='r')
#plt.grid()


for i in range(iniFig,endFig+1,1):
    temp = np.load('Total_Avs3_Mean_R_0%1s.npy'%(i))
    #plt.errorbar(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),yerr=temp.std(axis=1),color=colors[i])
    plt.plot(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),colors[i])
#    start_end[i,0] = temp.mean(axis=1)[19]
#    start_end[i,1] = temp.mean(axis=1)[0]
plt.ylim(-0.07, 0.02)
plt.grid()

##########################################################
### Total Number of games plot
##########################################################
#plt.subplot(1,4,2)
#plt.title('Number of games per trial')
#plt.xlabel('Trial')
#gamesT = np.zeros((N/binSize, numTest))
#
#plt.errorbar(np.arange(1,N/binSize+1)*binSize, gamesT.mean(axis=1),yerr=gamesT.std(axis=1),color='r')
#plt.grid()

##########################################################
### Win probability, epsilon decay and phi decay plots
##########################################################
#plt.subplot(1,4,3)
plt.figure(2)
plt.title('Avs3 transfer from Avs2 & Avs0 - NN Avs3 13-19-1')
plt.ylabel('Winning probability')
plt.xlabel('Trial')

#plt.plot(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi,'b')
#plt.plot((epsilonDecay**(np.arange(0,N)*numSamples)+bDecay)*(1-(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi)),'k')
#plt.plot((1-(epsilonDecay**(np.arange(0,N)*numSamples)+bDecay))*(1-(phiDecay**(np.arange(0,N)*numSamples)+bDecayPhi)),'m')

R_win = np.zeros((N/binSize, numTest))
Mean_start_end=np.zeros((10,2))
Var_start_end=np.zeros((10,2))
for i in range(iniFig,endFig+1,1):
    temp = np.load('Total_Avs3_Prob_win_0%1s.npy'%(i))
    #plt.errorbar(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),yerr=temp.std(axis=1),color=colors[i])
    plt.plot(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),colors[i])
    Mean_start_end[i,0] = temp.mean(axis=1)[19]
    Mean_start_end[i,1] = temp.mean(axis=1)[0]
    Var_start_end[i,0] = temp.std(axis=1)[19]
    Var_start_end[i,1] = temp.std(axis=1)[0]
    
plt.legend(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], loc=4)
plt.grid()

plt.figure(3,figsize=(17, 12))
plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,0], Var_start_end[:,0], linestyle='-.', marker='^')
##transfer_rate = np.arange(0,1.1,0.1)
##colors = ('ro','bo','go','co','mo','ko','yo','r*','b*','g*','c*','m*','k*','y*','r-.','b-.')
##for i in range(iniFig,endFig+1):
##    plt.plot(transfer_rate[i], start_end[i,0],colors[i],ms=8.0)

plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,1], Var_start_end[:,1], linestyle='-.', marker='^')
##transfer_rate = np.arange(0,1.1,0.1)
##colors = ('ro','bo','go','co','mo','ko','yo','r*','b*','g*','c*','m*','k*','y*','r-.','b-.')
##for i in range(iniFig,endFig+1):
##    plt.plot(transfer_rate[i], start_end[i,1],colors[i],ms=8.0)


plt.title('Avs3 transfer from Avs2 & Avs0 - NN Avs3 13-19-1', fontsize=20)
plt.ylabel('Winning probability', fontsize=18)
plt.xlabel('Transfer rate', fontsize=18)
plt.xlim(-0.05, 1.0)
plt.ylim(-0.1, 0.7)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.grid()
plt.savefig('Avs3WTpolicyAvs2NTAvs1WTAvs0 winningProb.png', bbox_inches='tight')

##########################################################
### Mean reinforcement
##########################################################

R_rew = np.zeros((N/binSize, numTest))
Mean_start_end=np.zeros((10,2))
Var_start_end=np.zeros((10,2))
for i in range(iniFig,endFig+1,1):
    temp = np.load('Total_Avs3_Mean_R_0%1s.npy'%(i))
    #plt.errorbar(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),yerr=temp.std(axis=1),color=colors[i])
    plt.plot(np.arange(1,N/binSize+1)*binSize, temp.mean(axis=1),colors[i])
    Mean_start_end[i,0] = temp.mean(axis=1)[19]
    Mean_start_end[i,1] = temp.mean(axis=1)[0]
    Var_start_end[i,0] = temp.std(axis=1)[19]
    Var_start_end[i,1] = temp.std(axis=1)[0]

plt.figure(4,figsize=(17, 12))
plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,0], Var_start_end[:,0], linestyle='-.', marker='^')
##transfer_rate = np.arange(0,1.1,0.1)
##colors = ('ro','bo','go','co','mo','ko','yo','r*','b*','g*','c*','m*','k*','y*','r-.','b-.')
##for i in range(iniFig,endFig+1):
##    plt.plot(transfer_rate[i], start_end[i,0],colors[i],ms=8.0)

plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,1], Var_start_end[:,1], linestyle='-.', marker='^')
##transfer_rate = np.arange(0,1.1,0.1)
##colors = ('ro','bo','go','co','mo','ko','yo','r*','b*','g*','c*','m*','k*','y*','r-.','b-.')
##for i in range(iniFig,endFig+1):
##    plt.plot(transfer_rate[i], start_end[i,1],colors[i],ms=8.0)


plt.title('Avs3 transfer from Avs2 & Avs0 - NN Avs3 13-19-1', fontsize=20)
plt.ylabel('Mean reinforcement', fontsize=18)
plt.xlabel('Transfer', fontsize=18)
plt.xlim(-0.05, 1.0)
#plt.ylim(0.1, 0.6)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.grid()
plt.savefig('Avs3WTpolicyAvs2NTAvs1WTAvs0 meanReinforcement.png', bbox_inches='tight')

##########################################################
### Tranfer rate and policy differences
##########################################################
#plt.subplot(1,4,4)
#plt.title('Transfer')
#plt.xlabel('Trial')
#TransferNum_mean = np.zeros((N/binSize, numTest))
#TransferEqual_mean = np.zeros((N/binSize, numTest))
#TransferRatio_mean = np.zeros((N/binSize, numTest))
#
#plt.errorbar(np.arange(1,N/binSize+1)*binSize, TransferNum_mean.mean(axis=1),yerr=TransferNum_mean.std(axis=1),color='r')
#plt.errorbar(np.arange(1,N/binSize+1)*binSize, TransferEqual_mean.mean(axis=1),yerr=TransferEqual_mean.std(axis=1),color='b')
#plt.errorbar(np.arange(1,N/binSize+1)*binSize, 100*TransferRatio_mean.mean(axis=1),yerr=100*TransferRatio_mean.std(axis=1),color='k')
#plt.grid()

plt.show()
