import socket
import Queue
import numpy as np

UDP_IP = "localhost"
UDP_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(('', UDP_PORT))

run = True

i = 0

global_array = []

first_fill = True

q = Queue.Queue(maxsize=100)

while run:
    
    myarray = []
    data, addr = sock.recvfrom(1024)
    
    split =  data.split(',')
    if (len(split) > 9):
        myarray.append([float(split[2]), float(split[3]), float(split[4]), float(split[6]), float(split[7]), float(split[8])])    
    
    if (first_fill):
        q.put_nowait(myarray)
        print q.qsize()
        if (q.full()):
            global_array = np.array(q.queue)
            print "send first array"
            first_fill = False
            # this is the first time
    else:
        i = i + 1
        print "i ", i
        ##rint q.qsize()
        q.get()
        q.put_nowait(myarray)
        
    if (i == 10):
        global_array = np.array(q.queue)
        print "send array with 10 new"
        #send arry
        global_array = q.queue
        i = 0
        