import pdb
import gradientDescent as gd
reload(gd)
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    """ Neural network with one hidden layer.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNet(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units,
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh

 For nonlinear classification (prediction of integer valued class labels)
   net = NeuralNetClassifier(ni,nh,no)
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x 1 (integer class labels
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   classes,Y,Z = net.use(Xtest,allOutputs=True)  # classes is nSamples x 1
"""
    def __init__(self,ni,nh,no):
        self.V = np.random.uniform(-0.1,0.1,size=(1+ni,nh))
        #self.V = np.random.uniform(-1,1,size=(1+ni,nh))
        self.W = np.random.uniform(-0.1,0.1,size=(1+nh,no))
        self.ni,self.nh,self.no = ni,nh,no
        self.standardize = None
        self.standardizeT = None
        
    def getSize(self):
        return (self.ni,self.nh,self.no)
    
    def train(self,X,T,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.standardize is None:
            self.standardize,_ = self.makeStandardize(X)
        X = self.standardize(X)
        X1 = self.addOnes(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.standardizeT is None:
            self.standardizeT,self.unstandardizeT = self.makeStandardize(T)
        T = self.standardizeT(T)

        # Local functions used by gradientDescent.scg()
        def pack(V,W):
            return np.hstack((V.flat,W.flat))
        def unpack(w):
            self.V[:] = w[:(self.ni+1)*self.nh].reshape((self.ni+1,self.nh))
            self.W[:] = w[(self.ni+1)*self.nh:].reshape((self.nh+1,self.no))
        def objectiveF(w):
            unpack(w)
            Y = np.dot(self.addOnes(np.tanh(np.dot(X1,self.V))), self.W)
            return 0.5 * np.mean((Y - T)**2)
        def gradF(w):
            unpack(w)
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )
            error = (Y - T) / (X1.shape[0] * T.shape[1])
            dV = np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
            dW = np.dot( Z1.T, error)
            return pack(dV,dW)

        scgresult = gd.scg(pack(self.V,self.W), objectiveF, gradF,
                           xPrecision = weightPrecision,
                           fPrecision = errorPrecision,
                           nIterations = nIterations,
                           ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)

    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def use(self,X,allOutputs=False):
        X1 = self.addOnes(self.standardize(X))
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = self.addOnes(Z)
        Y = self.unstandardizeT(np.dot( Z1, self.W ))
        return (Y,Z) if allOutputs else Y

    def getErrorTrace(self):
        return self.errorTrace
    
    def plotErrors(self):
        plt.plot(self.errorTrace)
        plt.ylabel("Train RMSE")
        plt.xlabel("Iteration")

    def addOnes(self,X):
        return np.hstack((np.ones((X.shape[0],1)), X))

    def makeStandardize(self,X):
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        def standardize(origX):
            return (origX - means) / stds
        def unStandardize(stdX):
            return stds * stdX + means
        return (standardize, unStandardize)

    def makeIndicatorVars(self,T):
        """ Assumes argument is N x 1, N samples each being integer class label """
        return (T == np.unique(T)).astype(int)

    def draw(self, inputNames = None, outputNames = None, gray = False):
        def isOdd(x):
            return x % 2 != 0

        W = [self.V, self.W]
        nLayers = 2

        # calculate xlim and ylim for whole network plot
        #  Assume 4 characters fit between each wire
        #  -0.5 is to leave 0.5 spacing before first wire
        xlim = max(map(len,inputNames))/4.0 if inputNames else 1
        ylim = 0
    
        for li in range(nLayers):
            ni,no = W[li].shape  #no means number outputs this layer
            if not isOdd(li):
                ylim += ni + 0.5
            else:
                xlim += ni + 0.5

        ni,no = W[nLayers-1].shape  #no means number outputs this layer
        if isOdd(nLayers):
            xlim += no + 0.5
        else:
            ylim += no + 0.5

        # Add space for output names
        if outputNames:
            if isOdd(nLayers):
                ylim += 0.25
            else:
                xlim += round(max(map(len,outputNames))/4.0)

        ax = plt.gca()

        x0 = 1
        y0 = 0 # to allow for constant input to first layer
        # First Layer
        if inputNames:
            #addx = max(map(len,inputNames))*0.1
            y = 0.55
            for n in inputNames:
                y += 1
                ax.text(x0-len(n)*0.2, y, n)
                x0 = max([1,max(map(len,inputNames))/4.0])

        for li in range(nLayers):
            Wi = W[li]
            ni,no = Wi.shape
            if not isOdd(li):
                # Odd layer index. Vertical layer. Origin is upper left.
                # Constant input
                ax.text(x0-0.2, y0+0.5, '1')
                for li in range(ni):
                    ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
                # cell "bodies"
                xs = x0 + np.arange(no) + 0.5
                ys = np.array([y0+ni+0.5]*no)
                ax.scatter(xs,ys,marker='v',s=1000,c='gray')
                # weights
                if gray:
                    colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
                xs = np.arange(no)+ x0+0.5
                ys = np.arange(ni)+ y0 + 0.5
                aWi = abs(Wi)
                aWi = aWi / np.max(aWi) * 50
                coords = np.meshgrid(xs,ys)
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                y0 += ni + 1
                x0 += -1 ## shift for next layer's constant input
            else:
                # Even layer index. Horizontal layer. Origin is upper left.
                # Constant input
                ax.text(x0+0.5, y0-0.2, '1')
                # input lines
                for li in range(ni):
                    ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
                # cell "bodies"
                xs = np.array([x0 + ni + 0.5]*no)
                ys = y0 + 0.5 + np.arange(no)
                ax.scatter(xs,ys,marker='>',s=1000,c='gray')
                # weights
                Wiflat = Wi.T.flatten()
                if gray:
                    colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wiflat >= 0)+0]
                xs = np.arange(ni)+x0 + 0.5
                ys = np.arange(no)+y0 + 0.5
                coords = np.meshgrid(xs,ys)
                aWi = abs(Wiflat)
                aWi = aWi / np.max(aWi) * 50
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                x0 += ni + 1
                y0 -= 1 ##shift to allow for next layer's constant input

        # Last layer output labels 
        if outputNames:
            if isOdd(nLayers):
                x = x0+1.5
                for n in outputNames:
                    x += 1
                    ax.text(x, y0+0.5, n)
            else:
                y = y0+0.6
                for n in outputNames:
                    y += 1
                    ax.text(x0+0.2, y, n)
        ax.axis([0,xlim, ylim,0])
        ax.axis('off')

class NeuralNetClassifier(NeuralNet):
    def __init__(self,ni,nh,no):
        #super(NeuralNetClassifier,self).__init__(ni,nh,no)
        NeuralNet.__init__(self,ni,nh,no-1)

    def train(self,X,T,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.standardize is None:
            self.standardize,_ = self.makeStandardize(X)
        X = self.standardize(X)
        X1 = self.addOnes(X)

        self.classes = np.unique(T)
        if self.no != len(self.classes)-1:
            raise ValueError(" In NeuralNetClassifier, the number of outputs must be one less than\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetClassifier()." % (self.no, len(self.classes)))
        T = self.makeIndicatorVars(T)

        # Local functions used by gradientDescent.scg()
        def pack(V,W):
            return np.hstack((V.flat,W.flat))
        def unpack(w):
            self.V[:] = w[:(self.ni+1)*self.nh].reshape((self.ni+1,self.nh))
            self.W[:] = w[(self.ni+1)*self.nh:].reshape((self.nh+1,self.no))
        def objectiveF(w):
            unpack(w)
            Y = np.dot(self.addOnes(np.tanh(np.dot(X1,self.V))), self.W)
            expY = np.exp(Y)
            denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
            Y = np.hstack((expY / denom, 1/denom))
            return -np.sum(T * np.log(Y)) 
        def gradF(w):
            unpack(w)
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )
            expY = np.exp(Y)
            denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
            Y = np.hstack((expY /denom , 1.0/denom))
            error = (T[:,:-1] - Y[:,:-1]) #/ (X1.shape[0] * T.shape[1])
            dV = -np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
            dW = -np.dot( Z1.T, error) 
            return pack(dV,dW)

        scgresult = gd.scg(pack(self.V,self.W), objectiveF, gradF,
                           xPrecision = weightPrecision,
                           fPrecision = errorPrecision,
                           nIterations = nIterations,
                           ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)

    def use(self,X,allOutputs=False):
        X1 = self.addOnes(self.standardize(X))
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = self.addOnes(Z)
        Y = np.dot( Z1, self.W )
        expY = np.exp(Y)
        denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
        Y = np.hstack((expY / denom, 1/denom))
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        return (classes,Y,Z) if allOutputs else classes

    def plotErrors(self):
        plt.plot(np.exp(-self.errorTrace))
        plt.ylabel("Train Likelihood")
        plt.xlabel("Iteration")

class NeuralNetQActionOutput(NeuralNet):
    def __init__(self,ni,nh,no,inputminmax):
        NeuralNet.__init__(self,ni,nh,no)
        print np.array(inputminmax).T
        self.standardize,_ = self.makeStandardize(np.array(inputminmax).T)

    def train(self,X,R,Q,Y,Ai,gamma=1,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.standardize is None:
            self.standardize,_ = self.makeStandardize(X)
        X = self.standardize(X)
        X1 = self.addOnes(X)

        # Local functions used by gradientDescent.scg()
        def pack(V,W):
            return np.hstack((V.flat,W.flat))
        def unpack(w):
            self.V[:] = w[:(self.ni+1)*self.nh].reshape((self.ni+1,self.nh))
            self.W[:] = w[(self.ni+1)*self.nh:].reshape((self.nh+1,self.no))
        def objectiveF(w):
            unpack(w)
            Y = np.dot(self.addOnes(np.tanh(np.dot(X1,self.V))), self.W)
            # Only calculate error for outputs corresponding to selected actions
            nSamples = Y.shape[0]
            Ybest = np.zeros((nSamples,1))
            for i in xrange(nSamples):
                Ybest[i,0] = Y[i,Ai[i,0]]
            return 0.5 *np.mean((R+gamma*Q-Ybest)**2)
        def gradF(w):
            unpack(w)
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )
            # Only calculate gradient with respect to error for outputs
            # corresponding to selected actions.
            # For each row of Y, select the column given by Ai
            nSamples = Y.shape[0]
            errorMat = np.zeros(Y.shape)
            for i in xrange(nSamples):
                j = Ai[i,0]
                errorMat[i,j] = -(R[i,0] + gamma*Q[i,0] - Y[i,j])
            errorMat /= nSamples
            dV = np.dot( X1.T, np.dot( errorMat, self.W[1:,:].T) * (1-Z**2))
            dW = np.dot( Z1.T, errorMat)
            return pack(dV,dW)

        if True:
            scgresult = gd.scg(pack(self.V,self.W), objectiveF, gradF,
                               xPrecision = weightPrecision,
                               fPrecision = errorPrecision,
                               nIterations = nIterations,
                               ftracep=True)
        else:
            scgresult = gd.steepest(pack(self.V,self.W), objectiveF, gradF,
                                    stepsize=0.01,xPrecision = weightPrecision,
                                    fPrecision = errorPrecision,
                                    nIterations = nIterations,
                                    ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)

    def use(self,X,allOutputs=False):
        if self.standardize:
            X1 = self.addOnes(self.standardize(X))
        else:
            X1 = self.addOnes(X)  # for NeuralNetQ if not trained yet
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = self.addOnes(Z)
        Y = np.dot( Z1, self.W )
        return (Y,Z) if allOutputs else Y


class NeuralNetQ(NeuralNet):
    def __init__(self,ni,nh,no,inputminmax):
        NeuralNet.__init__(self,ni,nh,no)
        self.standardize,_ = self.makeStandardize(np.array(inputminmax).T)

    def train(self,X,R,Q,Y,gamma=1,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.standardize is None:
            self.standardize,_ = self.makeStandardize(X)
        X = self.standardize(X)
        X1 = self.addOnes(X)

        # Local functions used by gradientDescent.scg()
        def pack(V,W):
            return np.hstack((V.flat,W.flat))
        def unpack(w):
            self.V[:] = w[:(self.ni+1)*self.nh].reshape((self.ni+1,self.nh))
            self.W[:] = w[(self.ni+1)*self.nh:].reshape((self.nh+1,self.no))
        def objectiveF(w):
            unpack(w)
            Y = np.dot(self.addOnes(np.tanh(np.dot(X1,self.V))), self.W)
            return 0.5 *np.mean((R+gamma*Q-Y)**2)
        def gradF(w):
            unpack(w)
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )
            nSamples = X1.shape[0]
            error = - (R + gamma * Q - Y) / nSamples
            dV = np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
            dW = np.dot( Z1.T, error)
            return pack(dV,dW)

        if True:
            scgresult = gd.scg(pack(self.V,self.W), objectiveF, gradF,
                               xPrecision = weightPrecision,
                               fPrecision = errorPrecision,
                               nIterations = nIterations,
                               ftracep=True)
        else:
            scgresult = gd.steepest(pack(self.V,self.W), objectiveF, gradF,
                                    stepsize=0.01,xPrecision = weightPrecision,
                                    fPrecision = errorPrecision,
                                    nIterations = nIterations,
                                    ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)

    def use(self,X,allOutputs=False):
        if not isinstance(X,np.ndarray):
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape((1,-1))
        if self.standardize:
            X1 = self.addOnes(self.standardize(X))
        else:
            X1 = self.addOnes(X)  # for NeuralNetQ if not trained yet
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = self.addOnes(Z)
        Y = np.dot( Z1, self.W )
        return (Y,Z) if allOutputs else Y


if __name__== "__main__":

    print "Classification Example"

    X = np.linspace(0,100,num=100).reshape((-1,1))
    T = np.array([1]*20+[2]*20+[3]*20+[4]*20+[5]*20).reshape((-1,1))
    nnet = NeuralNetClassifier(1,4,5)
    nnet.train(X,T,weightPrecision=1.e-8,errorPrecision=1.e-8,nIterations=1000)
    print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    (classes,y,Z) = nnet.use(X, allOutputs=True)
    print np.hstack((classes,T))
    
    print "Inputs"
    print X
    print "Targets"
    print T

    print "Hidden Outputs"
    print Z
    
    plt.ion()
    plt.figure()
    plt.clf()
    plt.subplot(3,2,1)
    nnet.plotErrors()
    plt.subplot(3,2,3)
    plt.plot(X,np.hstack((classes,T)))
    plt.subplot(3,2,5)
    #nnet.draw(['x1','x2'],['xor'])
    plt.plot(X,y)

    print "Regression Example"

    X = np.linspace(0,10,20).reshape((-1,1))
    T = 1.5 + 0.6 * X + 0.4 * np.sin(X)
    nnet = NeuralNet(1,5,1)
    nnet.train(X,T,weightPrecision=1.e-8,errorPrecision=1.e-8,nIterations=100)
    print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    (Y,Z) = nnet.use(X, allOutputs=True)
    
    print "Inputs"
    print X
    print "Targets"
    print T

    print np.hstack((T,Y))

    print "Hidden Outputs"
    print Z
    
    plt.subplot(3,2,2)
    nnet.plotErrors()
    plt.subplot(3,2,4)
    plt.plot(T,Y, 'o')
    plt.subplot(3,2,6)
    nnet.draw(['x'],['sine'])


    print "Reinforcement Learning Example"
    print "Three states, actions are left, stay, right. Walls at ends. Goal is middle"

    import random

    def epsilonGreedy(net,s,epsilon):
        if np.random.uniform() < epsilon:
            ai = random.randint(0,len(actions)-1)    # random action
        else:
            qs = [net.use((s,action)) for action in actions]
            ai = np.argmax(qs)                   # greedy action
        a = actions[ai]
        Q = net.use((s,a))
        return (a, Q)

def getSamples(net,actions,numSamples,epsilon,epsilonDecay):
    X = np.zeros((numSamples,net.getSize()[0]))
    R = np.zeros((numSamples,1))
    Qn = np.zeros((numSamples,1))
    Q = np.zeros((numSamples,1))

    # Initial state, action, and Q value
    s = initialState()
    a,q = epsilonGreedy(net,s,actions,epsilon)

    # Collect data from numSamples steps
    for step in xrange(numSamples):
        sn = nextState(s,a)        # Update state, sn from s and a
        rn = reinforcement(s,sn)   # Calculate resulting reinforcement
        an,qn = epsilonGreedy(net,sn,actions,epsilon) # Forward pass for time t+1

        X[step,:], R[step,0], Qn[step,0], Q[step,0] = s+[a], rn, qn, q

        s,a,q = sn,an,qn               # advance one time step
        epsilon *= epsilonDecay         # decay exploration probability
        
    return (X,R,Qn,Q,epsilon)


def makeSamples(net,N,epsilon):
        X = np.zeros((N,1))
        R = np.zeros((N,1))
        Q = np.zeros((N,1))
        Y = np.zeros((N,1))
        Ai = np.zeros((N,1))
        # Initial state  1 to 3
        s = random.randint(1,3)  
        # Select action using epsilon-greedy policy and get Q
        a,ai,q = epsilonGreedy(net,s,epsilon)
        for step in xrange(N):
            # Update state, s1 from s and a
            s1 = min(3,max(1, s + a))
            # Get resulting reinforcement
            r1 = 1 if s1 == 2 else 0
            # Select action for next step and get Q
            a1,ai1,q1 = epsilonGreedy(net,s1,epsilon)
            # Collect
            X[step,0], R[step,0], Q[step,0], Y[step,0], Ai[step,0] = s,r1,q1,q,ai
            # Shift state, action and action index by one time step
            s,a,ai = s1,a1,ai1
        return X,R,Q,Y,Ai

#    N = 100
#    epsilon = 1
#    net = NeuralNetQ(2,2,1,((1,3),(-1,1)))
#    for reps in range(3):
#        X,R,Q,Y,Ai = makeSamples(net,N,epsilon)
#        net.train(X,R,Q,Y,Ai,gamma=1,nIterations=1000,
#                  weightPrecision=0,errorPrecision=1e-5)
#        epsilon *= 0.99
#        # Print Q values
#        print "rep",reps
#        y = net.use(np.array([[1,-1],[1,0],[1,1], [2,-1],[2,0],[2,1], [3,-1],[3,0],[3,1]]))
#        print y
#        print np.array(('right','stay','left'))[np.argmax(y,axis=1)]

    
