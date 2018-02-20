
""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
from IPython.display import SVG
import matplotlib.pyplot as plt
import numpy as np
import time
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




data = json.load(open('C:\\Users\\muellersm\\Desktop\\71124z36_matrix.json'))
myarray = []
for i in range (0, 1400):
        myarray.append((([data["trial"]["frames"][i]["Mir_m00"]],[data["trial"]["frames"][i]["Mir_m01"]],[data["trial"]["frames"][i]["Mir_m02"]],[data["trial"]["frames"][i]["Mir_m10"]],[data["trial"]["frames"][i]["Mir_m11"]],[data["trial"]["frames"][i]["Mir_m12"]],[data["trial"]["frames"][i]["Mir_m20"]],[data["trial"]["frames"][i]["Mir_m21"]],[data["trial"]["frames"][i]["Mir_m22"]])))
myarray = np.array(myarray)


test1 =[]
test2 =[]
test3 =[]
test4 =[]
test5 =[]
test6 =[]
test7 =[]
test8 =[]
test9 =[]

for i in range (0, len(myarray)):
            test1.append(myarray[i][0])
for i in range (0, len(myarray)):
            test2.append(myarray[i][1])
for i in range (0, len(myarray)):
            test3.append(myarray[i][2])
for i in range (0, len(myarray)):
            test4.append(myarray[i][3])
for i in range (0, len(myarray)):
            test5.append(myarray[i][4])
for i in range (0, len(myarray)):
            test6.append(myarray[i][5])
for i in range (0, len(myarray)):
            test7.append(myarray[i][6])
for i in range (0, len(myarray)):
            test8.append(myarray[i][7])
for i in range (0, len(myarray)):
            test9.append(myarray[i][8])
            

plt.figure(1)
plt.plot (test1)
plt.figure(2)
plt.plot (test2)
plt.figure(3)
plt.plot (test3)
plt.figure(4)
plt.plot (test4)
plt.figure(5)
plt.plot (test5)
plt.figure(6)
plt.plot (test6)
plt.figure(7)
plt.plot (test7)
plt.figure(8)
plt.plot (test8)
plt.figure(9)
plt.plot (test9)

plt.show()
