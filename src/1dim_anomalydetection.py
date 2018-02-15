
""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
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




np.random.seed(1234)

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50


def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
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


def gen_wave():
    """ Generate a synthetic wave by adding up a few sine waves and some noise
    :return: the final wave
    """
#    t = np.arange(0.0, 10.0, 0.01)
#    wave1 = sin(2 * 2 * pi * t)
#    noise = random.normal(0, 0.1, len(t))
#    wave1 = wave1 + noise
#    print("wave1", len(wave1))
#    print("wave1", wave1)
#    wave2 = sin(2 * pi * t)
#    print("wave2", len(wave2))
#    t_rider = arange(0.0, 0.5, 0.01)
#    wave3 = sin(10 * pi * t_rider)
#    print("wave3", len(wave3))
#    insert = round(0.8 * len(t))
#    wave1[insert:insert + 50] = wave1[insert:insert + 50] + wave3
#    return wave1 + wave2

    data = json.load(open('..\\data\\71124z36.json'))
    myarray = []
    for i in range (0, 1400):
        myarray.append((data["trial"]["frames"][i]["Height"]))
    
    myarray = np.array(myarray)
    #myarray = np.diff(np.diff(myarray))
    
    print ("myarray: ", myarray)
#   myarray = np.flipud(myarray)
    
#    b, a = signal.butter(3, 0.05)
#    y = signal.filtfilt(b, a, myarray)
    return myarray


def z_norm(result):
    global stddev
    result_mean = result.mean()
    result_std = result.std()
    stddev = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    data = gen_wave()
    print (data.shape)
    print("Length of Data", len(data))

    # train data
    print ("Creating train data...")

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print ("Mean of train data : ", result_mean)
    print ("Train data shape  : ", result.shape)
    print (result)

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train)
    
#    for i in range (0, len(X_train)):
#        X_train [i] = X_train [i] + (i/10)
#    for i in range (0, len(y_train)):
#        y_train [i] = y_train [i] + (i/10)

    # test data
    print ("Creating test data...")

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)
    print ("result", result)
    print ("Mean of test data : ", result_mean)
    print ("Test data shape  : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))
    
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=sequence_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True)) 
    model.add(GaussianNoise(stddev))
    model.add(Dropout(0.2))
   

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))
    
   

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    
    """DOES NOT WORK"""
    #currently buggy due to keras issues 
    
    #plot_model(model, to_file='model.png')
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
    """DOES NOT WORK"""

    return model

def normalize(v):
    norm = v.max()
    if norm == 0: 
       return v
    return v / norm


def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print ('Loading data... ')
        # train on first 700 samples and test on next 300 samples (has anomaly)
        X_train, y_train, X_test, y_test = get_split_prep_data(0, 700, 500, 1000)
    else:
        X_train, y_train, X_test, y_test = data

    print ('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    try:
        print("Training...")
        model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=epochs, validation_split=0.05)
        print("Predicting...")
        predicttime_start = time.time()- global_start_time
        predicted = model.predict(X_test)
        predicttime_end = time.time() - global_start_time  
        pduration = predicttime_end - predicttime_start
        print ("Predict duration", pduration)
        
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        plt.figure(1)
        plt.subplot(311)
        #plt.title("Actual Test Signal w/Anomalies")
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
       # plt.title("Predicted Signal")
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
       # plt.title("Squared Error")
        mse = ((y_test - predicted) ** 2)
        mse = normalize(mse)
        
        #Anomaly Counting
#        anomaly_counter =0
#        i=0
#        while i < len(mse):
#            if mse [i] >= 0.5:
#                anomaly_counter += 1
#                while mse [i] >= 0.45: 
#                        i += 1
#            i += 1
#                        
#                            
#        print ("Anomalies detected:", anomaly_counter)
        
        plt.plot(mse, 'r')
        
        
#        x = np.arange(len(mse))
#        heights = [1] * len(mse)
#
#
#        cmap = plt.cm.rainbow
#        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
#
#        fig, ax = plt.subplots()
#        ax.bar( x, heights, width=1.0, color=cmap(norm(mse)))
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ax.set_ylim([0,10])
#        
#
#        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#        sm.set_array([])
#        fig.colorbar(sm)
        
        plt.show()
    except Exception as e:
        print("plotting exception")
        print (str(e))
    print ('Training duration (s) : ', time.time() - global_start_time)
    
    
    return model, y_test, predicted


run_network()


   
