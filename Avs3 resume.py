# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 12:36:09 2016

@author: Edwin
"""

import numpy as np
import matplotlib.pyplot as plt

i = 1

np.load('Avs3 WT Policy Avs0/Total/Total_Avs3_Mean_R_0%1s.npy'%(i))

file_path_meanR = ('Avs3 WT Policy Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs1 NT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs1 WT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs2 NT Avs1 NT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs2 NT Avs1 WT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs2 WT Avs1 NT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy',
             'Avs3 WT Policy Avs2 WT Avs1 WT Avs0/Total/Total_Avs3_Mean_R_0%1s.npy');

file_path_probWin = ('Avs3 WT Policy Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs1 NT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs1 WT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs2 NT Avs1 NT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs2 NT Avs1 WT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs2 WT Avs1 NT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy',
             'Avs3 WT Policy Avs2 WT Avs1 WT Avs0/Total/Total_Avs3_Prob_win_0%1s.npy');
             
colors = ('r','b','g','c','m','k','y','r','b','g','c','m','k','y','r','b')

N = 6000
binSize = 300

numTest = 15
iniFig = 0
endFig = 9

todas = False

plt.figure(1,figsize=(17, 12))
plt.plot([0, 0.9], [0.4008, 0.4008], 'k-.', lw=3)
plt.plot([0, 0.9], [0.533663, 0.533663], 'k--', lw=3)
for j in range(0,7):

    R_win = np.zeros((N/binSize, numTest))
    Mean_start_end=np.zeros((10,2))
    Var_start_end=np.zeros((10,2))
    for i in range(iniFig,endFig+1,1):
        temp = np.load(file_path_probWin[j]%(i))
        Mean_start_end[i,0] = temp.mean(axis=1)[19]
        Mean_start_end[i,1] = temp.mean(axis=1)[0]
        Var_start_end[i,0] = temp.std(axis=1)[19]
        Var_start_end[i,1] = temp.std(axis=1)[0]
    
    #plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,0], Var_start_end[:,0], color=colors[j], linestyle='-', marker='^')
    #plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,1], Var_start_end[:,1], color=colors[j], linestyle='-.', marker='^')
    plt.plot(np.arange(0,1.0,0.1), Mean_start_end[:,0], color=colors[j], linestyle='-', marker='^')
    if todas:
        plt.plot(np.arange(0,1.0,0.1), Mean_start_end[:,1], color=colors[j], linestyle='-.', marker='o')


#plt.title('T$_3$ transfer results', fontsize=20)
plt.ylabel('Winning probability', fontsize=18)
plt.xlabel('Transfer rate $\phi$', fontsize=18)
plt.xlim(-0.05, 1.0)
plt.ylim(-0.0, 0.7)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.grid()
if todas:
    lgd = plt.legend(('E$_8$ 6k','E$_8$ 18k','E$_9$','E$_9$','E$_{10}$','E$_{10}$','E$_{11}$','E$_{11}$','E$_{12}$','E$_{12}$','E$_{13}$','E$_{13}$','E$_{14}$','E$_{14}$','E$_{15}$','E$_{15}$'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
else:
    lgd = plt.legend(('E$_8$ 6k','E$_8$ 18k','E$_9$','E$_{10}$','E$_{11}$','E$_{12}$','E$_{13}$','E$_{14}$','E$_{15}$'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
plt.savefig('winProb.eps', dpi=300, format='eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
#= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

plt.figure(2,figsize=(17, 12))
plt.plot([0, 0.9], [-0.00669, -0.00669], 'k-.', lw=3)
plt.plot([0, 0.9], [0.01015, 0.01015], 'k--', lw=3)
for j in range(0,7):
    R_rew = np.zeros((N/binSize, numTest))
    Mean_start_end=np.zeros((10,2))
    Var_start_end=np.zeros((10,2))
    for i in range(iniFig,endFig+1,1):
        temp = np.load(file_path_meanR[j]%(i))
        Mean_start_end[i,0] = temp.mean(axis=1)[19]
        Mean_start_end[i,1] = temp.mean(axis=1)[0]
        Var_start_end[i,0] = temp.std(axis=1)[19]
        Var_start_end[i,1] = temp.std(axis=1)[0]
    

    #plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,0], Var_start_end[:,0], color=colors[j], linestyle='-', marker='^')
    #plt.errorbar(np.arange(0,1.0,0.1), Mean_start_end[:,1], Var_start_end[:,1], color=colors[j], linestyle='-.', marker='^')
    plt.plot(np.arange(0,1.0,0.1), Mean_start_end[:,0], color=colors[j], linestyle='-', marker='^')
    if todas:
        plt.plot(np.arange(0,1.0,0.1), Mean_start_end[:,1], color=colors[j], linestyle='-.', marker='o')
    
#plt.title('T$_3$ transfer results', fontsize=20)
plt.ylabel('Mean of reinforcement', fontsize=18)
plt.xlabel('Transfer rate $\phi$', fontsize=18)
plt.xlim(-0.05, 1.0)
#plt.ylim(0.1, 0.6)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.grid()
if todas:
    lgd = plt.legend(('E$_8$ 6k','E$_8$ 18k','E$_9$','E$_9$','E$_{10}$','E$_{10}$','E$_{11}$','E$_{11}$','E$_{12}$','E$_{12}$','E$_{13}$','E$_{13}$','E$_{14}$','E$_{14}$','E$_{15}$','E$_{15}$'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
else:
    lgd = plt.legend(('E$_8$ 6k','E$_8$ 18k','E$_9$','E$_{10}$','E$_{11}$','E$_{12}$','E$_{13}$','E$_{14}$','E$_{15}$'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('meanReinf.eps', dpi=300, format='eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

#plt.show()
