
""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.


Rebuild for live tracking by Enrico Kaack
"""
from IPython.display import SVG
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers import GaussianNoise
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import arange, sin, pi, random
import json
from scipy import signal
import math
import plotly.plotly as py
import plotly.graph_objs as go
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.colors
import pandas as pd
import sys


np.random.seed(1234)

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50


class Predictor:
    model
    X_test = []
    y_test = []
    min = 0
    max = 0
    
    def __init__(self, model, min, max):
        self.model = model
        self.min = min
        self.max = max
    
    def predictForFrame(self, data):
        X_test = data[:, :-1]
        self.y_test = data[:, -1]
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
        
        z= []
        start = 0        
        for i in range (start, len(predicted)+start):
          z.append(abs(predicted [i][0])+abs(predicted [i][1])+abs(predicted [i][2])+abs(predicted [i][3])+abs(predicted [i][4])+abs(predicted [i][5]))
            
        z = np.array(z)
            
        signal =[]
        for i in range (start, len(predicted)+start):
            signal. append(abs(y_test [i][0])+abs(y_test [i][1])+abs(y_test [i][2])+abs(y_test [i][3])+abs(y_test [i][4])+abs(y_test [i][5]))

 
        mse = ((signal - z) ** 2)        
        
         # generate threshold
        max = 0
        min = 0
        for i in range (0, len(mse)):
            if (mse[i] > max):
                max = mse[i]
            if (mse[i] < min):
                min = mse[i]
            
        print("threshold", min, max)

        

class Trainer:
    trainData = []
    X_train = []
    y_train = []
    min = 0
    max = 0
    
    def trainOnPrerecorded(self, filePath, model, trainStart, trainEnd):
        self.loadDataFromFile(filePath)
        self.prepareData(trainStart, trainEnd)
        self.train(model)
        self.predictToGenerateThreshold(model)
        return model, self.min, self.max

    def loadDataFromFile(self, filePath):
        assert filePath != '', "file path incorrect"
        with open(filePath) as f:
            lines = f.readlines()
            print("Lines in CSV: ", len(lines))
            i = 0
            trainData = []
            for line in lines:
                i = i+1
                split =  line.split(',')
                if (len(split) > 9):
                    trainData.append([float(split[2]), float(split[3]), float(split[4]), float(split[6]), float(split[7]), float(split[8])])
        
        trainData = np.array(trainData)
        trainData = np.reshape(trainData,(trainData.shape[0],6))
        self.trainData = trainData
        return trainData
    
    def prepareData(self, trainStart, trainEnd):
        result = []
        for index in range(trainStart, trainEnd - sequence_length):
            result.append(self.trainData[index: index + sequence_length])
        result = np.array(result) 
        result, result_mean = self.z_norm(result)
    
        print ("Mean of train data : ", result_mean)
        print ("Train data shape  : ", result.shape)
    
        train = result[trainStart:trainEnd, :]
        np.random.shuffle(train)  # shuffles in-place
        X_train = train[:, :-1]
        y_train = train[:, -1]
        self.X_train, self.y_train = dropin(X_train, y_train)
        
    def z_norm(self, result):
    
        global stddev
        result_mean = result.mean()
        result_std = result.std()
        stddev = result.std()
        result -= result_mean
        result /= result_std
        return result, result_mean
    
    def dropin(self, X, y):
        """ inverse of dropout --> add values
        http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
        :param X: Each row is a training sequence
        :param y: Tne target we train and will later predict
        :return: new augmented X, y
        """
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_hat = []
        y_hat = []
        for i in range(0, len(X)):
            for j in range(0, np.random.random_integers(0, random_data_dup)):
                X_hat.append(X[i, :])
                y_hat.append(y[i])
                
                
        return np.asarray(X_hat), np.asarray(y_hat)
    
    def train(self, model):
        try:
            print("Training...")
            model.fit(
                self.X_train, self.y_train,
                batch_size=batch_size, epochs=epochs, validation_split=0.05)
        except KeyboardInterrupt:
            print("exception")
        
        


    def predictToGenerateThreshold(self, model):
        predicted = model.predict(self.X_train)
        
        z= []
        start = 0
        
        for i in range (start, len(predicted)+start):
            z.append(abs(predicted [i][0])+abs(predicted [i][1])+abs(predicted [i][2])+abs(predicted [i][3])+abs(predicted [i][4])+abs(predicted [i][5]))
            
        z = np.array(z)
            
        signal =[]
        for i in range (start, len(predicted)+start):
            signal. append(abs(self.y_train [i][0])+abs(self.y_train [i][1])+abs(self.y_train [i][2])+abs(self.y_train [i][3])+abs(self.y_train [i][4])+abs(self.y_train [i][5]))
            
        signal = np.array(signal)
        
        mse = ((signal - z) ** 2)
        
        # generate threshold
        max = 0
        min = 0
        for i in range (0, len(mse)):
            if (mse[i] > max):
                max = mse[i]
            if (mse[i] < min):
                min = mse[i]
        self.max = max
        self.min = min
        print("threshold", min, max)


def build_model():          
    model = Sequential()
    layers = {'input': 6,  'test1': 1, 'hidden1': 64, 'hidden2': 9, 'hidden3': 64, 'test2': 1, 'output': 6}
    
    model.add(LSTM(
            input_shape=(sequence_length - 1,layers['input']),
            units=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(GaussianNoise(stddev))
    
    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))
    
    
       
    
    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(
            units=layers['output']))
    model.add(Activation("linear"))
    
    start = time.time()
    model.compile(loss="logcosh", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)

    return model





def normalize(v):
    norm = v.max()
    if norm == 0: 
       return v
    return v / norm


model = build_model()

trainer = Trainer()
model, min, max = trainer.trainOnPrerecorded('..\\data\\hit.csv', model, 200, 1000)
predictor = Predictor(model, min, max)