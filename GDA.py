# author: Matthew P. Burruss
# date: 5/20/2019
# description: A program to perform Gaussian Discriminant Analysis (GDA)
# GDA is solves classification problems where the input features
# are continuous-real valued using the generative format p(x|y) essentially
# choosing the best model y that maximizes the conditional probability above
# #
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
from scipy.stats import bernoulli

class GDA:
    def findMLE(self):
        # calculate MLEs multinomail ditribution P(Y={0...k-1})
        for i in range(self.M):
            self.phis[ Y[i] ] = self.phis[ Y[i] ] + 1
        self.phis = np.divide(self.phis,self.M)
        # calculate mean MLE for MVNs
        occurrences = np.zeros(shape=(self.k,1))
        for i in range(self.M):
            self.means[Y[i],:] = self.means[Y[i],:] + X[i,:]
            occurrences[Y[i]] = occurrences[Y[i]] + 1
        self.means = np.divide(self.means,occurrences)
        # calculate cov MLE for MVNs
        occurrences = np.zeros(shape=(self.k,1))
        for i in range(self.M):
            self.covs[:,:,Y[i]] = self.covs[:,:,Y[i]] + np.transpose(np.add(X[i,:],-1*self.means[Y[i],:]))*np.add(X[i,:],-1*self.means[Y[i],:]) 
            occurrences[Y[i]] = occurrences[Y[i]] + 1
        for j in range(self.k):
            self.covs[:,:,j] = np.divide(self.covs[:,:,j],occurrences[j])
        self.rv_bernoullis = []
        self.rv_MVNs = []
        for i in range(self.k):
            self.rv_bernoullis.append(bernoulli(p=self.phis[i]))
            self.rv_MVNs.append(multivariate_normal(mean=self.means[i,:], cov=self.covs[:,:,i]))
    def createModel(self):
        print("######################################")
        print("Gaussian Discriminant Analysis")
        print("######################################")
        print("Size of data: %d" %(self.M))
        print("Number of input variables: %d" %(self.numberOfFeatures))
        print("Number of output categories: %d" %(self.k))
        print("######################################")
        print("Calculating model parameters...")
        self.findMLE()
        if(self.verbose):
            print("Multinomial parameter estimates (MLEs p1...pk): ")
            print(self.phis)
            print("MVN mean parameter estimate (mu1...muk): ")
            print(self.means)
            print("MVN covariance matrix parameter estimates (sig1...sigk): ")
            print(self.covs)
        print("Finished calculating model parameters...")
        print("######################################")

    def predict(self,x):
        probabilitiesNotNormalized = np.zeros(shape=(1,self.k))
        for i in range(self.k):
            probabilitiesNotNormalized[:,i] = self.rv_MVNs[i].pdf(x)*self.rv_bernoullis[i].pmf(1)
        print(probabilitiesNotNormalized)
    # takes in input matrix X where each row is a feature vector and output Y of k categories
    def __init__(self,X,Y,k,verbose=False):
        self.X = X      # input
        self.Y = Y      # output category
        self.M = len(Y) # number of input-output pairs
        self.k = k      # number of categories
        self.verbose = verbose # determine how much to tell the user
        self.numberOfFeatures = X.shape[1]   # number of input variables
        self.phis = np.zeros(shape=(k,1))
        print(self.phis)
        self.means = np.zeros(shape=(k,self.numberOfFeatures))
        self.covs = np.zeros(shape=(self.numberOfFeatures,self.numberOfFeatures,k))
        self.createModel()
def visualizeMVN():
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-2,2)
    ax.set_ylim3d(-2,2)
    ax.set_zlim3d(0,1)
    X, Y = np.mgrid[-2:2:.01, -2:2:.01] # create a 2D mesh grid
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean=[0, 0], cov=[[0.2, 0], [0, 0.2]])
    Z = rv.pdf(pos)
    ax.contour3D(X,Y,Z,100,cmap='viridis')
    ax2.contour(X,Y,Z)
    plt.show()

def predict(x,rv1,rv2,rv3):
    print(rv1.pdf(x)*rv3.pmf(0))
    print(rv2.pdf(x)*rv3.pmf(1))
        
    

X = np.matrix([[4,3],[4.2,3.3],[4.4,3.0],[1.0,2.2],[.9,3.3],[0.2,2.5]])
Y = [0,0,0,1,1,1]
#rv1,rv2,rv3 = GDAa(X,Y)
model = GDA(X,Y,k=2)
model.predict([.9,3])
#x = np.array([1,2])
#predict([4,3],rv1,rv2,rv3)
