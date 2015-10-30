# -*- coding: utf-8 -*-
"""
Echo State Networks
http://minds.jacobs-university.de/mantas
http://organic.elis.ugent.be/
http://www.scholarpedia.org/article/Echo_state_network
Mantas Lukoševičius "A Practical Guide to Applying Echo State Networks"
Konrad Stanek "Reservoir computing in 
nancial forecasting with committee methods"
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression

# generate Mackey-Glass time series
#import Oger
#train_signals = Oger.datasets.mackey_glass(sample_len=10000, tau=200, n_samples=1)
#savetxt(r'data\MackeyGlass_t200.txt', np.array(train_signals[0]).reshape(-1,1))

np.random.seed(42)
      
class ESN(object):

    def __init__(self, resSize=500, rho=0.9, cr=0.05, leaking_rate=0.2, W=None):
        """
        :param resSize: reservoir size
        :param rho: spectral radius
        :param cr: connectivity ratio
        :param leaking_rate: leaking rate
        :param W: predefined ESN reservoir
        """
        self.resSize = resSize
        self.leaking_rate = leaking_rate

        if W is None:
            # generate the ESN reservoir
            N = resSize * resSize
            W = np.random.rand(N) - 0.5
            zero_index = np.random.permutation(N)[int(N * cr * 1.0):]
            W[zero_index] = 0
            W = W.reshape((self.resSize, self.resSize))
            # Option 1 - direct scaling (quick&dirty, reservoir-specific):
            #self.W *= 0.135 
            # Option 2 - normalizing and setting spectral radius (correct, slow):
            print 'ESN init: Setting spectral radius...',
            rhoW = max(abs(linalg.eig(W)[0]))
            print 'done.'
            W *= rho / rhoW
        else:
            assert W.shape[0] == W.shape[1] == resSize, "reservoir size mismatch"
        self.W = W

    def __init_states__(self, X, initLen, reset_state=True):

        # allocate memory for the collected states matrix
        self.S = np.zeros((len(X) - initLen, 1 + self.inSize + self.resSize))
        if reset_state:
            self.s = np.zeros(self.resSize)
        s = self.s.copy()

        # run the reservoir with the data and collect S
        for t, u in enumerate(X):
            s = (1 - self.leaking_rate) * s + self.leaking_rate *\
                                np.tanh(np.dot(self.Win, np.hstack((1, u))) +\
                                np.dot(self.W, s))
            if t >= initLen:
                self.S[t-initLen] = np.hstack((1, u, s))
        if reset_state:
            self.s = s

    def fit(self, X, y, lmbd=1e-6, initLen=100, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d), where
        :         t - length of time series, d - dimensionality.
        :param y : array-like, shape (t,). Target vector relative to X.
        :param lmbd: regularization lambda
        :param initLen: Number of samples to wash out the initial random state
        :param init_states: False allows skipping states initialization if
        :                   it was initialized before (with same X).
        :                   Useful in experiments with different targets.
        """
        assert len(X) == len(y), "input lengths mismatch."
        self.inSize =  1 if np.ndim(X) == 1 else X.shape[1]
        if init_states:
            print("ESN fit_ridge: Initializing states..."),
            self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
            self.__init_states__(X, initLen)
            print("done.")
        self.ridge = Ridge(alpha=lmbd, fit_intercept=False,
                               solver='svd', tol=1e-6)
        self.ridge.fit(self.S, y[initLen:])
        return self

    def fit_proba(self, X, y, lmbd=1e-6, initLen=100, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d)
        :param y : array-like, shape (t,). Target vector relative to X.
        :param lmbd: regularization lambda
        :param initLen: Number of samples to wash out the initial random state
        :param init_states: see above
        """
        assert len(X) == len(y), "input lengths mismatch."
        self.inSize =  1 if np.ndim(X) == 1 else X.shape[1]
        if init_states:        
            print("ESN fit_proba: Initializing states..."),
            self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
            self.__init_states__(X, initLen)
            print("done.")
        self.logreg = LogisticRegression(C=1/lmbd, penalty='l2',
                                         fit_intercept=False, 
                                         solver='liblinear')
        self.logreg.fit(self.S, y[initLen:])
        return self

    def predict(self, X, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
        :param init_states: see above
        """
        if init_states:        
            # assume states initialized with training data and we continue from there.
            self.__init_states__(X, 0, reset_state=False)
        y = self.ridge.predict(self.S)
        return y
        
    def predict_proba(self, X, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
        :param init_states: see above
        """
        if init_states:
            # assume states initialized with training data and we continue from there.
            self.__init_states__(X, 0, reset_state=False)
        y = self.logreg.predict_proba(self.S)
        return y[:,1]


if __name__ == '__main__':
    
    # load the data
    data = np.loadtxt(r'data\MackeyGlass_t50.txt')
    data = MinMaxScaler(feature_range=(-0.3, 0.3)).fit_transform(data)
    noise = 0.1 # add n.d. noise
    data += noise * np.std(data) * np.random.randn(len(data))
    
    trainLen = 8000
    testLen = 1000
    dif = 46  # prediction horizon >=1
    
    X = data[0:trainLen]
    y = data[dif:trainLen+dif]
    y_p = map(lambda x: 1 if x else 0, data[dif:trainLen+dif] > data[0:trainLen])
    Xtest = data[trainLen:trainLen+testLen]
    ytest = data[trainLen+dif:trainLen+testLen+dif]
    ytest_p = map(lambda x: 1 if x else 0, data[trainLen+dif:trainLen+testLen+dif] > data[trainLen:trainLen+testLen])
    
    resSize = 500
    rho = 0.9  # spectral radius
    cr = 0.05 # connectivity ratio
    leaking_rate = 0.2 # leaking rate
    lmbd = 1e-6 # regularization coefficient
    initLen = 100
    esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)
    
    esn.fit(X, y, initLen=initLen, lmbd=lmbd)
    esn.fit_proba(X, y_p, initLen=initLen, lmbd=lmbd, init_states=False)
    y_predicted = esn.predict(Xtest)
    y_predicted_p = esn.predict_proba(Xtest, init_states=False)
    
    # compute metrics
    errorLen = testLen
    mse = mean_squared_error(ytest[0:errorLen], y_predicted[0:errorLen])
    auc = roc_auc_score(ytest_p[0:errorLen], y_predicted_p[0:errorLen])
    fpr, tpr, _ = roc_curve(ytest_p[0:errorLen], y_predicted_p[0:errorLen])
    y_predicted_lab = np.zeros(len(y_predicted_p))
    y_predicted_lab[ y_predicted_p >= 0.5] = 1
    acc = accuracy_score(ytest_p[0:errorLen], y_predicted_lab[0:errorLen])
    
    print("Ridge regression MSE = {}".format(mse))
    print("Logistic regression AUC = {}, Accuracy = {}.".format(auc, acc))
    
    
    ####################################################################### 
    # Plot of signals
    
    plt.figure(10).clear()
    plt.plot(data[0:1000])
    plt.title('A sample of data')
    
    plt.figure(1).clear()
    plt.plot( ytest, 'g', y_predicted, 'b')
    plt.title('Predicting {} steps ahead. MSE = {:7.5f}'.format(dif,mse))
    plt.legend(['Target signal', 'Ridge regression'], loc="upper right")
    
    plt.figure(2).clear()
    plt.plot(esn.S[0:200, 2:20])
    plt.title('Some reservoir activations $\mathbf{x}(n)$')
    
    # Plot of a ROC curve
    plt.figure(3).clear()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    plt.show()
    
    
    #######################################################################
    # Esperiments with prediction horizon
    
    difs = range(1,100)
    mse_a = np.empty(len(difs))
    auc_a = np.empty(len(difs))
    acc_a = np.empty(len(difs))
    for i,dif in enumerate(difs):
        y = data[dif:trainLen+dif]
        y_p = map(lambda x: 1 if x else 0, y > data[0:trainLen])
        ytest = data[trainLen+dif:trainLen+testLen+dif]
        ytest_p = map(lambda x: 1 if x else 0, ytest > data[trainLen:trainLen+testLen])
       
        esn.fit(X, y, initLen=initLen, lmbd=lmbd)
        esn.fit_proba(X, y_p, initLen=initLen, lmbd=lmbd, init_states=False)
        y_predicted = esn.predict(Xtest)
        y_predicted_p = esn.predict_proba(Xtest, init_states=False)
            
        # compute metrics
        errorLen = testLen
        mse = mean_squared_error(ytest[0:errorLen], y_predicted[0:errorLen])
        auc = roc_auc_score(ytest_p[0:errorLen], y_predicted_p[0:errorLen])
        fpr, tpr, _ = roc_curve(ytest_p[0:errorLen], y_predicted_p[0:errorLen])
        y_predicted_lab = np.zeros(len(y_predicted_p))
        y_predicted_lab[ y_predicted_p >= 0.5] = 1
        acc = accuracy_score(ytest_p[0:errorLen], y_predicted_lab[0:errorLen])
        
        print("dif = {} ({} to go):".format(dif, len(difs)-i-1))
        print("\tRidge regression MSE = {}".format(mse))
        print("\tLogistic regression AUC = {}, Accuracy = {}.".format(auc, acc)) 
        
        mse_a[i] = mse
        auc_a[i] = auc
        acc_a[i] = acc
        
    plt.figure(4).clear()
    plt.plot(difs, mse_a, 'b')
    plt.xlim([difs[0], difs[-1]])
    plt.title('MSE as a function of prediction horizon')
    plt.ylabel('MSE')
    
    plt.figure(5).clear()
    plt.plot(difs, auc_a, 'r')
    plt.xlim([difs[0], difs[-1]])
    plt.title('AUC as a function of prediction horizon')
    plt.ylabel('AUC')
    
    plt.figure(6).clear()
    plt.plot(difs, acc_a, 'g')
    plt.xlim([difs[0], difs[-1]])
    plt.title('Accuracy as a function of prediction horizon')
    plt.ylabel('Accuracy')
    
    plt.show()