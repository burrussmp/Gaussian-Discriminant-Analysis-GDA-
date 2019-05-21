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
import math

class GDA:
    def findMLE(self):
        # calculate MLEs multinomail ditribution P(Y={0...k-1})
        for i in range(self.M):
            self.phis[ self.Y[i] ] = self.phis[ self.Y[i] ] + 1
        self.phis = np.divide(self.phis,self.M)
        # calculate mean MLE for MVNs
        occurrences = np.zeros(shape=(self.k,1))
        for i in range(self.M):
            self.means[self.Y[i],:] = self.means[self.Y[i],:] + self.X[i,:]
            occurrences[self.Y[i]] = occurrences[self.Y[i]] + 1
        self.means = np.divide(self.means,occurrences)
        # calculate cov MLE for MVNs
        occurrences = np.zeros(shape=(self.k,1))
        for i in range(self.M):
            ls = np.squeeze(self.covs[:,:,self.Y[i]])
            rs = np.transpose(np.add(self.X[i,:],-1*self.means[self.Y[i],:]))*np.add(self.X[i,:],-1*self.means[self.Y[i],:])
            sum = ls + rs
            sum = np.expand_dims(sum,3)
            self.covs[:,:,self.Y[i]] = sum
            occurrences[self.Y[i]] = occurrences[self.Y[i]] + 1
        for j in range(self.k):
            self.covs[:,:,j] = np.divide(self.covs[:,:,j],occurrences[j])
        self.rv_bernoullis = []
        self.rv_MVNs = []
        for i in range(self.k):
            self.rv_bernoullis.append(bernoulli(p=self.phis[i]))
            try:
                self.rv_MVNs.append(multivariate_normal(mean=self.means[i,:], cov=self.covs[:,:,i],allow_singular=False))
            except:
                self.rv_MVNs.append(multivariate_normal(mean=self.means[i,:], cov=self.covs[:,:,i],allow_singular=True))
                print("WARNING: Singular matrix for model %d consider increasing dataset for model" %i)
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
        probability = np.zeros(shape=(1,self.k))
        p_x = 0.0
        model = 0
        best = -np.inf
        for i in range(self.k):
            probability[:,i] = self.rv_MVNs[i].pdf(x)*self.rv_bernoullis[i].pmf(1)
            if (probability[:,i]>best):
                best = probability[:,i]
                model = i
            p_x = p_x + self.rv_MVNs[i].pdf(x)*self.rv_bernoullis[i].pmf(1) 
        probability = np.divide(probability,p_x)
        return model

    def validate(self,validationX,validationY):
        correct = 0.0
        categoriesTested = np.zeros(shape=(self.k,1))
        categoriesRight = np.zeros(shape=(self.k,1))
        for i in range(len(validationY)):
            predictedCategory = self.predict(validationX[i,:])
            categoriesTested[validationY[i]] = categoriesTested[validationY[i]] + 1
            if (predictedCategory == validationY[i]):
                correct = correct + 1.0
                categoriesRight[validationY[i]] = categoriesRight[validationY[i]] + 1
        print("######################################")
        print("VALIDATION")
        print("N = %d" %(len(validationY)))
        print("Categories validated")
        for i in range(self.k):
            print("Category %d: validated=%d | correct = %d" %(i,categoriesTested[i],categoriesRight[i]))
        print("Total accuracy (% correct): {:.2f}".format(correct/len(validationY)))
        print("######################################")

    # takes in input matrix X where each row is a feature vector and output Y of k categories
    def __init__(self,X,Y,k,verbose=False):
        self.allInputs = X
        self.k = k # number of categories
        self.Y = Y      # output category
        self.M = len(Y) # number of input-output pairs
        self.verbose = verbose # determine how much to tell the user
        self.reset(X=self.allInputs)
    
    def reset(self,X):
        self.X = np.copy(X)      # input
        self.numberOfFeatures = len(self.X[0])   # number of input variables
        self.phis = np.zeros(shape=(self.k,1))
        self.means = np.zeros(shape=(self.k,self.numberOfFeatures))
        self.covs = np.zeros(shape=(self.numberOfFeatures,self.numberOfFeatures,self.k))
        self.createModel()

    def visualizeMVN(self,variables=None):
        if (variables == None):
            v1 = int(input("Enter first variable you would like to visualize:   "))
            v2 = int(input("Enter second variable you would like to visualize:  "))
            variables = [v1, v2]
        if (len(variables) != 2):
            print("Visualization only works for 2 variables currently and %d variables were specified..." %(len(variables)))
            return
        if (variables[0] > len(self.X[0]) or variables[1] > len(self.X[0])):
            print("Only %d variables were used to construct the model" %(len(variables)))
            return
        print("Creating visual of GDA for variables %d and %d" %(variables[0],variables[1]))
        print("Recalculating MLEs for model parameters")
        self.reset(self.X[:,variables])
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        min = np.full(shape=(self.numberOfFeatures,1),fill_value=np.inf)
        max = np.full(shape=(self.numberOfFeatures,1),fill_value=-np.inf)
        for i in range(self.M):
            for j in range(self.numberOfFeatures):
                if (min[j] > self.X[i,j]): min[j] = self.X[i,j]
                if (max[j] < self.X[i,j]): max[j] = self.X[i,j]    
        x, y = np.mgrid[min[0]:max[0]:100j, min[1]:max[1]:100j] # create a 2D mesh grid
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        for i in range(self.k):
            z = self.rv_MVNs[i].pdf(pos)
            ax.contour3D(x,y,z,100,cmap='viridis')
            ax2.contour(x,y,z)
        # plot data points
        Y = np.squeeze(self.Y)
        ax2.scatter([self.X[Y==0,0]],[self.X[Y==0,1]],marker = '.')
        ax2.scatter([self.X[Y==1,0]],[self.X[Y==1,1]],marker = 'x')
        # reset parameters
        print("Recalculating MLEs for model parameters")
        self.reset(self.allInputs)
        plt.show(block=False)

# returns dataset in matrix fomat
def readDataSet(name='wdbc.data'):
    data= []
    with open('wdbc.data','r') as filestream:
        for line in filestream:
            currentline = np.array(line.split(","))
            data.append(currentline)
    dataMatrix = np.empty(shape=(len(data),len(data[0])),dtype=object)
    for i in range(len(data)):
        dataMatrix[i,:] = data[i]
    return dataMatrix

def createTrainingAndValidation(X,Y,sizeOfTraining=0.8):
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    trainingIndices=indices[0:math.floor(sizeOfTraining*len(Y))]
    validationIndices = indices[math.ceil(sizeOfTraining*len(Y)):]
    trainingX = X[trainingIndices,:]
    trainingY = Y[trainingIndices]
    validationX = X[validationIndices,:]
    validationY = Y[validationIndices]
    return trainingX,trainingY,validationX,validationY

# data => unformatted data matrix
# inputcol => array of indices representing the columns for real-valued inputs
# outputcol => array of indices representing the output
def parseDataSet(data,inputCol,outputCol):
    Y = np.empty(shape=(len(data),len(outputCol)),dtype=int)
    X = np.zeros(shape=(len(data),len(inputCol)))
    X = data[:,inputCol].astype(dtype=float)
    categories = np.unique(data[:,outputCol])
    for i in range(len(Y)):
        Y[i] = np.where(categories==data[i,outputCol])
    return X,Y
  
data = readDataSet()
X,Y = parseDataSet(data,inputCol=range(2,32),outputCol=[1])
trainingX,trainingY,validationX,validationY = createTrainingAndValidation(X,Y,sizeOfTraining=0.8)
model = GDA(trainingX,trainingY,k=2)
model.validate(validationX,validationY)
while 1:
    model.visualizeMVN()

