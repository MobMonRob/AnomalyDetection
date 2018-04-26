
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
import queue
import socket
import h5py
from keras.models import load_model



np.random.seed(1234)

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50
FFT = False

class Predictor:
    X_test = []
    y_test = []
    anomaly = False
    anomalyCounter = 0
    
    def __init__(self, model, min, max):
        self.model = model
        self.min = min
        self.max = max
        
    def z_norm(self, result):
        global stddev ## evtl fixen
        result_mean = result.mean()
        result_std = result.std()
        stddev = result.std()
        stmean = result_mean
        result -= result_mean
        result /= result_std
        return result
    
    def predictForFrame(self, data):
        result = []
        for index in range(0, len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        result = np.array(result) 
        result = self.z_norm(result)

        X_test = result[:, :-1]
        y_test = result[:, -1]
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))
        
        if (FFT):
            print("start FFT")
            X_test = self.X_test
            for i in range(0, len(X_test)):
                outer = X_test[i, :, :]
                for j in range(0, outer.shape[1]):
                    fft = (np.fft.fft(outer[:, [j]]))
                    outer[:,[j]] = fft
                X_test[i] = outer
            self.X_test = X_test
            print("finished FFT ", X_test.shape )
        
        predicted = model.predict(self.X_test)
        #predicted = np.reshape(predicted, (predicted.size,))
        
        
        z= []
        start = 0        
        for i in range (0, len(predicted)):
          z.append(abs(predicted [i][0])+abs(predicted [i][1])+abs(predicted [i][2])+abs(predicted [i][3])+abs(predicted [i][4])+abs(predicted [i][5]))
            
        z = np.array(z)
            
        signal =[]
        for i in range (0, len(predicted)):
            signal.append(abs(y_test [i][0])+abs(y_test [i][1])+abs(y_test [i][2])+abs(y_test [i][3])+abs(y_test [i][4])+abs(y_test [i][5]))

 
        mse = ((signal - z) ** 2)        
        
         # generate threshold
        max = 0
        min = 0
        for i in range (0, len(mse)):
            if (mse[i] > max):
                max = mse[i]
            if (mse[i] < min):
                min = mse[i]
        
        # alarm if out of threshold
        if (max > self.max or min < self.min):
            if not self.anomaly:
                self.anomaly = True
                self.anomalyCounter = self.anomalyCounter + 1
        else:
            if self.anomaly:
                self.anomaly = False
        print(self.anomalyCounter, 'Error = ', max)

class Trainer:
    trainData = []
    X_train = []
    y_train = []
    testDate = []
    X_test = []
    y_test = []
    min = 0
    max = 0
    
    def prepareOnPrerecorded(self, filePath):
        self.loadDataFromFile(filePath)
        trainStart = 0
        trainEnd = int(len(self.trainData)*0.9)
        testStart = trainEnd + 1
        testEnd = len(self.trainData)
        print("Train: " + str(trainStart) + "-" + str(trainEnd) + "   Test: " + str(testStart) + "-" + str(testEnd))
        self.prepareData(trainStart, trainEnd, testStart, testEnd)
    
    def trainGroundLevel(self, model):
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
        print(len(trainData))
        self.trainData = trainData
    
    def prepareData(self, trainStart, trainEnd, testStart, testEnd):
        result = []
        for index in range(trainStart, trainEnd - sequence_length):
            result.append(self.trainData[index: index + sequence_length])
        result = np.array(result) 
        result, result_mean = self.z_norm(result)
    
        print ("Train data shape  : ", result.shape)
    
        train = result[trainStart:trainEnd, :]
        np.random.shuffle(train)  # shuffles in-place
        X_train = train[:, :-1]
        y_train = train[:, -1]
        self.X_train, self.y_train = self.dropin(X_train, y_train)
        
        if (FFT):
            print("start FFT")
            X_train = self.X_train
            for i in range(0, len(X_train)):
                outer = X_train[i, :, :]
                for j in range(0, outer.shape[1]):
                    fft = (np.fft.fft(outer[:, [j]]))
                    outer[:,[j]] = fft
                X_train[i] = outer
            self.X_train = X_train
            print("finished FFT ", X_train.shape )
        
        ##Test data to generate threshold
        result = []
        for index in range(testStart, testEnd - sequence_length):
            result.append(self.trainData[index: index + sequence_length])
        result = np.array(result) 
        result, result_mean = self.z_norm(result)
    
        print ("Test data shape  : ", result.shape)
    
        test = result[0:testEnd-testStart, :]
        self.X_test = test[:, :-1]
        self.y_test = test[:, -1]
        
        if (FFT):
            print("start FFT test data")
            X_test = self.X_test
            for i in range(0, len(X_test)):
                outer = X_test[i, :, :]
                for j in range(0, outer.shape[1]):
                    fft = (np.fft.fft(outer[:, [j]]))
                    outer[:,[j]] = fft
                X_train[i] = outer
            self.X_test = X_test
            print("finished FFT test", X_test.shape )
        
        
    def z_norm(self, result):
        global stddev
        global stmean
        result_mean = result.mean()
        result_std = result.std()
        stddev = result.std()
        stmean = result_mean
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
        predicted = model.predict(self.X_test)
        
        z= []
        start = 0
        
        for i in range (start, len(predicted)+start):
            z.append(abs(predicted [i][0])+abs(predicted [i][1])+abs(predicted [i][2])+abs(predicted [i][3])+abs(predicted [i][4])+abs(predicted [i][5]))
            
        z = np.array(z)
            
        signal =[]
        for i in range (start, len(predicted)+start):
            signal. append(abs(self.y_test [i][0])+abs(self.y_test [i][1])+abs(self.y_test [i][2])+abs(self.y_test [i][3])+abs(self.y_test [i][4])+abs(self.y_test [i][5]))
            
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


class UDP_Detector:
    
    # size of frame
    frame_size = 0
    
    # displacement size
    disp_size = 0
    
    # port used for UDP connection
    port = 0
    
    run = True
    
    sock = None
    
    # array used for the prediction
    predict_array = []
    
    def __init__(self, port, frame_size, disp_size, predictor):
        self.frame_size = frame_size
        self.port = port
        self.disp_size = disp_size
        self.predictor = predictor
       
    def predict(self):
        self.predictor.predictForFrame(self.predict_array)
    
    def start(self):
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        #open socket
        self.sock.bind(('', self.port))
        
        print("Socket opened")
        
        first_fill = True
        
        self.predict_array = []
        
        # init Q according to frame size
        frameQ = queue.Queue(maxsize=self.frame_size)
        
        self.run = True
        
        i = 0
        
        while self.run:
            measurement = []
            data, addr = self.sock.recvfrom(1024)
            
            # decode bytestream to string
            data = data.decode("utf-8")

            # split CSV
            split =  data.split(",")
            
            # fill measurement array
            if (len(split) > 9):
                measurement = [float(split[2]), float(split[3]), float(split[4]), float(split[6]), float(split[7]), float(split[8])]    
            
                # fill Q initially
                if (first_fill):
                    
                    # fill Q
                    frameQ.put_nowait(measurement)
                    
                    if (frameQ.full()):
                        
                        # dump Q to numpy array
                        self.predict_array = np.array(frameQ.queue)
                        
                        self.predict()
                        
                        # next measurement will pop the first element of the Q
                        first_fill = False
                else:
                    i = i + 1
                    
                    # pop first element of the Q
                    frameQ.get()
                    
                    frameQ.put_nowait(measurement)
            
                if (i == self.disp_size): # Q updated according to disp_size
                                        
                    #dump Q to numpy
                    self.predict_array = np.array(frameQ.queue)
                    
                    # call predictor
                    self.predict()
                    #print ("started prediction with ", self.disp_size, " new elements in frame")
                    
                    i = 0
        def stop():
            self.run = False

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



trainer = Trainer()


## train + save
#model = build_model()
#trainer.prepareOnPrerecorded('..\\data\\train_parcour.csv')
#model, min, max = trainer.trainGroundLevel(model)
#model.save('trained-train_parcour.h5', overwrite=True)
#with open("thresholds.txt", 'w') as file:
#    file.write(str(min) + "#" + str(max))


##load + predict
model = load_model('trained-train_parcour.h5')
f = open('thresholds_parcour.txt', 'r')
input = f.read().split('#')
min = float(input[0])
max = float(input[1])

predictor = Predictor(model, min, max)
detector = UDP_Detector(8888, 500, 50, predictor)
detector.start()