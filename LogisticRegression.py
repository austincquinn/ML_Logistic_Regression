import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class BinaryClassifier():

    def __init__(self, train_data, train_target):
        #Data is loaded from sklearn.datasets.load_breast_cancer 
        #train_data is training feature data and train_target is your train label data.
        
        #add new column of 1's to training data for w_0 as bias
        newCol = np.ones((train_data.shape[0], 1))
        train_data = np.append(train_data, newCol, 1)
        #normalize data
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        
        X_train, X_test, y_train, y_test=train_test_split(train_data,train_target,test_size=0.1)

        X_train = np.array(X_train)
        y_train = np.array([y_train]).T
        X_test = np.array(X_test)
        y_test = np.array([y_test]).T
        self.X_test = X_test
        self.y_test = y_test
        self.X = X_train
        self.y = y_train
        self.X_batch = 0
        self.y_batch = 0
        self.bestLoss = float("inf")
        self.bestW = None
        self.bestLambda = None
        self.bestAlpha = None
        self.clf = None

    def iterate_minibatches(self, inputs, targets, batchsize):
        #helps generate mini-batches

        assert inputs.shape[0] == targets.shape[0]
        for start_idx in range(0, inputs.shape[0], batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def logistic_training(self, alpha, lam, nepoch, epsilon):
        """Training process of logistic regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        i.e., if error < epsilon, training stops.
        Implementation includes 3-fold validation and mini-batch GD"""

        ep = 10**-10

        #3-fold cross validation
        data = np.append(self.X, self.y, 1)
        np.random.shuffle(data)
        split1, split2, split3, = np.array_split(data, 3)
        for cv in range(3):
            if cv is 0:
                validation = split1
                training = np.append(split2, split3, 0)
            elif cv is 1:
                validation = split2
                training = np.append(split1, split3, 0)
            elif cv is 2:
                validation = split3
                training = np.append(split1, split2, 0)
            
            X_train = training[:,:-1]
            y_train = np.array([training[:,-1]]).T
            X_val = validation[:,:-1]
            y_val = np.array([validation[:,-1]]).T
            
            mini_batch_size = 32

            curAlpha = alpha[0]
            curLambda = lam[0]
            while (curAlpha <= alpha[1]):        #mult min alpha by some number until max alpha
                while (curLambda <= lam[1]):        #mult min lam by some number until max lam
                    
                    w = np.random.rand(1, X_train.shape[1])

                    for epoch in range(nepoch):
                        for batch in self.iterate_minibatches(X_train, y_train, mini_batch_size):
                            self.X_batch, self.y_batch = batch
                            
                            #forward propagation
                            pred = 1/(1 + np.exp(-1 * np.dot(w,self.X_batch.T)))

                            #loss is sum of cross entropy (my loss is kinda more like cost)
                            loss = -(1/self.X_batch.shape[0]) * np.sum((self.y_batch * np.log(pred + ep) + (1-self.y_batch) * np.log(1-pred + ep))) + (1/2) * curLambda * np.sum(w**2)
                            #cost = (loss / mini_batch_size) + (1/2) * curLambda * np.sum(w**2)

                            #backward propagation
                            w_grad = w
                            #w_grad = -(1/mini_batch_size) * np.dot(self.X_batch.T, (pred - self.y_batch).T) + curLambda * w
                            w_grad = (1/self.X_batch.shape[0]) * np.sum(np.dot((self.y_batch - pred), self.X_batch)) + curLambda * w
                            
                            #Adagrad
                            w = w - (curAlpha / np.sqrt(np.sum(w_grad**2))) * w_grad
                            #Vanilla Grad
                            #w = w - curAlpha * w_grad

                            #ceiling to prevent overflow
                            if (loss > 10000):
                                break

                            # Comparing loss to epsilon
                            if loss < epsilon:
                                break
                    
                    cvPred = 1/(1 + np.exp(-1 * np.dot(w,X_val.T)))
                    cvLoss = (1/X_val.shape[0])*np.sum((y_val * np.log(cvPred + ep) + (1-y_val) * np.log(1-cvPred + ep))) + (1/2) * curLambda * np.sum(w**2)

                    if cvLoss < self.bestLoss:
                        self.bestLoss, self.bestW, self.bestAlpha, self.bestLambda = cvLoss, w, curAlpha, curLambda

                    curLambda *= 1.1
                curAlpha *= 1.1
            
        #train with all data
        mini_batch_size = 32
        w = self.bestW
        curLambda = self.bestLambda
        curAlpha = self.bestAlpha
        for epoch in range(nepoch*3):
            for batch in self.iterate_minibatches(self.X, self.y, mini_batch_size):
                self.X_batch, self.y_batch = batch
                
                #forward propagation
                pred = 1/(1 + np.exp(-1 * np.dot(w,self.X_batch.T)))

                #loss is sum of cross entropy
                loss = (1 / self.X_batch.shape[0]) * np.sum((self.y_batch * np.log(pred + ep) + (1-self.y_batch) * np.log(1-pred + ep))) + (1/2) * curLambda * np.sum(w**2)
                #currently don't do anything with cost
                #cost = (loss / mini_batch_size) + (1/2) * curLambda * np.sum(w**2)

                #backward propagation
                w_grad = w
                #w_grad = -(1/mini_batch_size) * np.dot(self.X_batch.T, (pred - self.y_batch).T) + curLambda * w
                w_grad = -(1 / self.X_batch.shape[0]) * np.sum(np.dot((self.y_batch - pred), self.X_batch)) + curLambda * w
                
                #Adagrad
                w = w - (curAlpha / np.sqrt(np.sum(w_grad**2))) * w_grad
                #Vanilla Grad
                #w = w - curAlpha * w_grad

                #ceiling to prevent overflow
                if (loss > 10000):
                    break

                # Comparing loss to epsilon
                if loss < epsilon:
                    break
                
        self.bestW = w

    def logistic_testing(self, testX):
        """TestX should be a numpy array
        Uses trained weight and bias to compute the predicted y values,
        Predicted y values should be 0 or 1. returns the numpy array in shape n*1"""

        #add new column of 1's to training data for w_0 as bias
        newCol = np.ones((testX.shape[0], 1))
        testX = np.append(testX, newCol, 1)
        #normalize data
        scaler = StandardScaler()
        testX = scaler.fit_transform(testX)

        y = np.ones(testX.shape[0])
        y = 1/(1 + np.exp(-1 * np.dot(self.bestW,testX.T)))
        y = (y < 0.5).astype(int)
        y = y.T
        return y

    def svm_training(self, gamma, C):
        """Uses sklearn's built-in GridSearchCV and SVM methods for comparison with logistic regression model"""

        parameters = {'gamma': gamma, 'C': C}
        #defaults RBF
        svc = svm.SVC()
        self.clf = GridSearchCV(svc, parameters)
        self.clf.fit(self.X, self.y)



    def svm_testing(self, testX):
        """TestX should be a numpy array
        Uses trained weight and bias to compute the predicted y values,
        Predicted y values should be 0 or 1. returns the numpy array in shape n*1"""
        
        #add new column of 1's to training data for w_0 as bias
        newCol = np.ones((testX.shape[0], 1))
        testX = np.append(testX, newCol, 1)
        #normalize data
        scaler = StandardScaler()
        testX = scaler.fit_transform(testX)
        y = self.clf.predict(testX)
        y = (y > 0.5).astype(int)
        y = np.array([y]).T
        return y


#main testing
dataset = load_breast_cancer(as_frame=True)
#Dataset is divided into 90% and 10%, 90% for you to perform k-fold validation and 10% for testing
train_data = dataset['data'].sample(frac=0.9, random_state=0) # random state is a seed value
train_target = dataset['target'].sample(frac=0.9, random_state=0) # random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)

model = BinaryClassifier(train_data, train_target)

# Compute the time to do grid search on training logistic
logistic_start = time.time()
model.logistic_training([10**-10, 10], [10e-10, 1e10], 400, 10**-6)
logistic_end = time.time()
# Compute the time to do grid search on training SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()