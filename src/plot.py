# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:38:45 2018

@author: enrico-kaack
"""
import matplotlib.pyplot as plt

myarray = []
with open('..\\data\\Messung_1.csv') as f:
    lines = f.readlines()   
    for line in lines:
        split =  line.split(',')
        if (len(split) > 9):
            myarray.append([split[2], split[3], split[4], split[6], split[7], split[8]])
                
    plotArray = []
    for item in myarray:
        plotArray.append(item[0:3])
        
    plt.plot(plotArray)
    plt.show()