# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:23:36 2018

@author: enrico-kaack
"""

import matplotlib.pyplot as plt
import scipy.fftpack as fpack
import numpy as np

import platform
system = platform.system()

if system == 'Darwin':
    dir_spacer = '/'
elif system == 'Windows':
    dir_spacer = '\\'    

# Number of samplepoints
N = 750
# sample spacing
T = 1.0 / 800.0

myarray = []
with open('..' + dir_spacer + 'data' + dir_spacer + 'Messung_1.csv') as f:
    lines = f.readlines()   
    for line in lines:
        split =  line.split(',')
        if (len(split) > 9):
            myarray.append([split[2], split[3], split[4], split[6], split[7], split[8]])
         
    normal = myarray[750:]
    transformed = fpack.fft(normal)
    print(transformed)
    
    
    yf = fpack.fft(normal)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()
    
    
    #plotArray = []
    #for item in myarray:
      #  plotArray.append(item[0:3])
        
    #plt.plot(transformed)
    #plt.show()