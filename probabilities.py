# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:17:15 2016

@author: Edwin
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmFunc(steepness, midpoint, x):
    return 1 / ( 1 + np.exp( -steepness*(x-midpoint) ) )

numSamples = 400
nTrials = 25000

initialPhi = 1.0
finalPhi = 1.e-2
bDecayPhi = initialPhi - 1.0
phiDecay = np.exp(np.log(finalPhi-bDecayPhi)/(nTrials*numSamples))

initialEpsilon = 1.0
finalEpsilon = 1.e-2
bDecay = initialEpsilon - 1.0
epsilonDecay = np.exp(np.log(finalEpsilon-bDecay)/(nTrials*numSamples))

phiSeries = (1.0)*phiDecay**(np.arange(0,nTrials)*numSamples)
epsilonSeries1 = epsilonDecay**(np.arange(0,nTrials)*numSamples)
exploration = epsilonSeries1*(1-phiSeries)

explotation = (1-epsilonSeries1)*(1-phiSeries)

plt.plot(phiSeries,'r')
plt.plot(exploration,'--b')
plt.plot(explotation,'-.k')
plt.ylabel('Probabilities')
plt.xlabel('Trials')
plt.grid()
plt.legend(('transfer','exploration','expotation'),'best')
plt.savefig('probabilities.png', bbox_inches='tight')

steepness, midpoint = 2, 7
x = np.arange(0,225)**(0.5)
y = 1 / ( 1 + np.exp( -steepness*(x-midpoint) ) )

plt.figure(2)
plt.plot(x,y)
plt.ylabel('Similarity$')
plt.xlabel('x')
plt.grid()
plt.savefig('Similarity.png', bbox_inches='tight')

plt.show()


