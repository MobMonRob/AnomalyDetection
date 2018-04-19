# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:28:20 2018

@author: enrico-kaack
"""

import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.sendto(b'test', ("172.16.58.23", 8888))