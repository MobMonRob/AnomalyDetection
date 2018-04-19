# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:23:36 2018

@author: enrico-kaack
"""

import matplotlib.pyplot as plt
import scipy.fftpack as fpack

myarray = []
with open('..\\data\\Messung_1.csv') as f:
    lines = f.readlines()   
    for line in lines:
        split =  line.split(',')
        if (len(split) > 9):
            myarray.append([split[2], split[3], split[4], split[6], split[7], split[8]])
         
    normal = myarray[:750]
    transformed = fpack.fft(normal)
    
    #plotArray = []
    #for item in myarray:
      #  plotArray.append(item[0:3])
        
    plt.plot(transformed)
    plt.show()