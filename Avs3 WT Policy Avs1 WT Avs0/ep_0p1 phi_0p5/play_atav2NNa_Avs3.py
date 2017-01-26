# -*- coding: utf-8 -*-
"""
Created on Sun Sep 02 21:54:02 2012

@author: Edwin
"""
import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as nn

actions = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])
                    
actionsM = np.array([[ 0, 0],
                     [-3, 0],
                     [ 0, 3],
                     [ 3, 0],
                     [ 0,-3]])

movesB = np.array([[ 0, 0],
                   [ 0,-1],
                   [ 0, 1],
                   [-1, 0],
                   [-1,-1],
                   [-1, 1],
                   [ 1, 0],
                   [ 1,-1],
                   [ 1, 1]])

epsilon_BX = (0.1, 0.2, 0.2)
etaX = (1.0, 1.0, 1.0)

BoardSize = 36

Rval = 1

###############################################################################
## Neural Network for the old task
#Nin = 9
#Nhidden = 7
#Nout = 1
nnet_OT = nn.NeuralNetQ(9, 10, 1, ((0,35) ,(0,35) ,(0.030303,1) ,(0.030303,1), (0,1),(0,1),(0,1),(0,1),(0,1)) )

#nnet_OT.W = np.load('Avs1_W_atav2NN_03.npy')
#nnet_OT.V = np.load('Avs1_V_atav2NN_03.npy')
###############################################################################
def jugar(posA, posBX,acc):

    posAn = np.zeros((1,2),int)
    posBXn = np.zeros((len(posBX),2),int)

    fin = 0
    rew = 0

    for i in range(len(posBX)):
        
        if np.random.random() < epsilon_BX[i]:
            tmp = np.random.permutation(9)
            c = tmp[0,]
        else:
            cref1 = 2*(BoardSize -1) / 3
            fref1 = -2
            bfr1 = np.around((posA[0,0] - fref1)*etaX[i]+fref1)
            bcr1 = np.around((posA[0,1] - cref1)*etaX[i]+cref1)
            dist = np.power(bfr1 - (posBX[i,0] + movesB[:,0]), 2) + np.power(bcr1 - (posBX[i,1] + movesB[:,1]), 2)
            c = np.argmin(dist)
        
        posBXn[i,:] = np.array([[posBX[i,0]+ movesB[c,0] ,posBX[i,1]+ movesB[c,1]]])
      
        if posBXn[i,0] < 2: posBXn[i,0] = posBXn[i,0]+1
        if posBXn[i,0] > (BoardSize-1)-0: posBXn[i,0] = posBXn[i,0]-1
        if posBXn[i,1] < 0: posBXn[i,1] = posBXn[i,1]+1
        if posBXn[i,1] > BoardSize-1: posBXn[i,1] = posBXn[i,1]-1

    ###########################################################################

#    if np.random.random() < epsilon_B1:
#        tmp = np.random.permutation(9)
#        c = tmp[0,]
#    else:
#        cref1 = 2*(BoardSize -1) / 3
#        fref1 = -2
#        bfr1 = np.around((posA[0,0] - fref1)*eta1+fref1)
#        bcr1 = np.around((posA[0,1] - cref1)*eta1+cref1)
#        dist = np.power(bfr1 - (posB1[0,0] + movesB[:,0]), 2) + np.power(bcr1 - (posB1[0,1] + movesB[:,1]), 2)
#        c = np.argmin(dist)
#    
#    posB1n = np.array([[posB1[0,0]+ movesB[c,0] ,posB1[0,1]+ movesB[c,1]]])
#  
#    if posB1n[0,0] < 2: posB1n[0,0] = posB1n[0,0]+1
#    if posB1n[0,0] > (BoardSize-1)-0: posB1n[0,0] = posB1n[0,0]-1
#    if posB1n[0,1] < 0: posB1n[0,1] = posB1n[0,1]+1
#    if posB1n[0,1] > BoardSize-1: posB1n[0,1] = posB1n[0,1]-1

    ###########################################################################

    posAn = np.array([[posA[0,0] + actionsM[acc,0] ,posA[0,1] + actionsM[acc,1] ]])

    if posAn[0,0] < 0: posAn[0,0] = posAn[0,0]+3
    if posAn[0,0] > (BoardSize-1): posAn[0,0] = posAn[0,0]-3
    if posAn[0,1] < 0: posAn[0,1] = posAn[0,1]+3
    if posAn[0,1] > (BoardSize-1): posAn[0,1] = posAn[0,1]-3

    ###########################################################################

    if any((np.abs(posAn[0,0]-posBX[:,0])<=2) * (np.abs(posAn[0,1]-posBX[:,1])<=2)):
        rew = -Rval
        fin = 1
    if ( (0 <= posAn[0,1] <= 8) or (27 <= posAn[0,1] <= 35) ) and posAn[0,0] == 1:
        rew = -Rval
        fin = 1
    if ( posAn[0,0] == 1 and  (8 < posAn[0,1] < 27) ):
        rew = Rval
        fin = 1
    
    return rew, posAn, posBXn, fin

################################################################
# Agent attacker distance
################################################################
def distanceABG(pA,pB):
    
    distABG = np.zeros((1,len(pB)*2))
    
    for i in range(len(pB)):
        distAB1 = np.sqrt( np.power(pB[i,0] - pA[0,0], 2) + np.power(pB[i,1] - pA[0,1], 2) )
        distABG[0,2*i] = distAB1 if distAB1 != 0 else 0.1
        distB1Goal = np.sqrt( np.power(pB[i,0] - 2, 2) + np.power(pB[i,1] - 9, 2) )
        distABG[0,2*i+1] = distB1Goal if distB1Goal != 0 else 0.1

    return distABG


###################################################################
def sigmFunc(steepness, midpoint, x):
    return 1 / ( 1 + np.exp( -steepness*(x-midpoint) ) )

###################################################################
def actionSelection(nnet_NT,pA,pB,epsilon):

    distABG = distanceABG(pA,pB)

    if np.random.uniform() < epsilon:
        ai = np.random.randint(0,len(actions))
    else:
        qs = np.zeros((5,1))
        qs = nnet_NT.use( np.hstack((pA.repeat(len(actions),axis=0), 1/distABG.repeat(len(actions),axis=0), actions)) )
        ai = np.argmax(qs)
        
    Q = nnet_NT.use( np.hstack((pA, 1/distABG, actions[ai,:].reshape(1,5))) )

    return Q, ai

###################################################################

def iniState(NumOpponents):

    bc = np.random.random_integers(0,35,NumOpponents)
    bf = np.floor(np.abs((np.random.normal(0, 8, NumOpponents))))+3
    rows = bf > 35
    bf[rows] = 3
    
    ac = np.random.random_integers(0,11)
    af = np.random.random_integers(1,11)
    
    while any((np.abs(af*3+1-bf)<=2) * (np.abs(ac*3+1-bc)<=2)):
        bc = np.random.random_integers(0,35,NumOpponents)
        bf = np.floor(np.abs((np.random.normal(0, 8, NumOpponents))))+3
        rows = bf > 35
        bf[rows] = 3

        ac = np.random.random_integers(0,11)
        af = np.random.random_integers(1,11)

    return np.array([[af*3+1, ac*3+1]]), np.rot90(np.vstack((bf,bc)))
    
###################################################################

def ppal(nnet_NT,numSamples,epsilon,epsilonDecay,phi,phiDecay):

    transfCont = 0
    transfDiff = 0
    
    NumOpponents = 3
    
    X = np.zeros((numSamples,2*(NumOpponents+1) + len(actions)))
    R = np.zeros((numSamples,1))
    Qn = np.zeros((numSamples,1))
    Q = np.zeros((numSamples,1))
    
    step = 0

    while (step < numSamples):
    
        fin = 0

        posA, posB = iniState(NumOpponents)
        
        #q, ai = epsilonGreedy(nnet,posA,posB1,posB2,epsilon)
        q, ai, transfCont, transfDiff = actionSelection(nnet_NT,posA,posB,epsilon,phi,transfCont,transfDiff)

        while (fin == 0 and step < numSamples):

            rew, posAn, posBn, fin = jugar(posA, posB, ai)
            #qn, ani = epsilonGreedy(nnet,posAn,posB1n,posB2n,epsilon)
            qn, ani, transfCont, transfDiff = actionSelection(nnet_NT,posAn,posBn,epsilon,phi,transfCont,transfDiff)
            
            INVdistABG = 1/distanceABG(posA,posB)

            X[step,:] = np.hstack((posA, INVdistABG, actions[ai,:].reshape(1,5)))
            R[step,0] = rew
            Qn[step,0] = qn
            Q[step,0] = q
            
            if fin == 1: Qn[step,0] = 0
                        
            ai, posA, posB, q = ani, posAn, posBn, qn
            step += 1
            epsilon *= epsilonDecay
            phi *= phiDecay

    return (X,R,Qn,Q,epsilon,phi,transfCont,transfDiff)

###################################################################

def plotBoard(A, B):
    plt.clf()    
    plt.plot(A[0,1], A[0,0],'ro', A[0,1]+1,A[0,0],'ro', A[0,1]-1,A[0,0],'ro', A[0,1], A[0,0]+1,'ro', A[0,1]+1,A[0,0]+1,'ro', A[0,1]-1,A[0,0]+1,'ro', A[0,1], A[0,0]-1,'ro', A[0,1]+1,A[0,0]-1,'ro', A[0,1]-1,A[0,0]-1,'ro')

    for i in range(len(B)):
        plt.plot(B[i,1],B[i,0],'bo')
    
    plt.plot(9,2,'ko', 26,2,'ko')
    plt.axis([-1,BoardSize,-1,BoardSize])
    plt.grid()
    plt.draw()

###############################################################################
##############  P P A L #######################################################
###############################################################################
   
def play(testNum):
    #plt.ion()
    numGames = 2000
    NumOpponents = 3
    
    Nin = 2*(NumOpponents+1) + len(actions)
    Nhidden = 19
    Nout = 1
    nnet_NT = nn.NeuralNetQ(Nin, Nhidden, Nout, ((0,35), (0,35), (0.030303,1), (0.030303,1), (0.030303,1), (0.030303,1), (0.030303,1), (0.030303,1), (0,1),(0,1),(0,1),(0,1),(0,1)) )

    nnet_NT.W = np.load('Avs3_W_atav2NN_0%1s.npy'%(testNum))
    nnet_NT.V = np.load('Avs3_V_atav2NN_0%1s.npy'%(testNum))
    
    win = np.zeros((numGames,1))

    epsilon = 0.0
    
    for numGame in range(numGames):

        posA, posB = iniState(NumOpponents)
        q, ai = actionSelection(nnet_NT,posA,posB,epsilon)
    
        fin = 0
    
        while fin == 0:
            
            rew, posAn, posBn, fin = jugar(posA, posB, ai)
            q, ani = actionSelection(nnet_NT, posAn, posBn,epsilon)
    
    #        if (fin == 1 and rew == -1): print("Perdio")
            if (fin == 1 and rew == 1): win[numGame,0] = 1#print("Gano")
    #        plotBoard(posAn, posBn)
    #        time.sleep(0.1)
    
            ai, posA, posB = ani, posAn, posBn

    #np.save('Avs3_win_0%1s'%(testNum), win)
    print testNum, win.mean()
    
    return win.mean()


if __name__== "__main__":
    
    tNum = 1
    winT = np.zeros((15,1))
    while (tNum < 16):
        winT[tNum-1,0] = play(tNum)
        tNum += 1
    print winT.mean()