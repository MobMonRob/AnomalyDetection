import numpy as np
import queue
import socket

class UDP_Detector:
    
    # size of frame
    frame_size = 0
    
    # displacement size
    disp_size = 0
    
    # port used for UDP connection
    port = 0
    
    run = True
    
    # array used for the prediction
    predict_array = []
    
    def __init__(self, port, frame_size, disp_size, predictor):
        self.frame_size = frame_size
        self.port = port
        self.disp_size = disp_size
        self.predictor = predictor
        
    def predict(self):
        self.predictor.predictForFrame(self.predict_array)
        
    def startWithPrerecordedData(self, offlinePath):
        f = open(offlinePath, 'r')
        lines = f.readlines()
        print('number of data points in file ', len(lines))
        first_fill = True
        self.predict_array = []
        
        # init Q according to frame size
        frameQ = queue.Queue(maxsize=self.frame_size)
        
        
        for line in lines:
            split = line.split(',')
            measurement = []
            measurement.append([float(split[2]), float(split[3]), float(split[4]), float(split[6]), float(split[7]), float(split[8])])    
            
            # fill Q initially
            if (first_fill):
                
                # fill Q
                frameQ.put_nowait(measurement)
                
                if (frameQ.full()):
                    
                    print ("Q initially filled")
                    
                    # dump Q to numpy array
                    self.predict_array = np.array(frameQ.queue)
                    
                    # call the predictor
                    ##self.predict()
                    print ("started prediction with initial Q")
                    
                    # next measurement will pop the first element of the Q
                    first_fill = False
            else:
                i = i + 1
                
                # pop first element of the Q
                frameQ.get()
                
                frameQ.put_nowait(measurement)
        
            if (i == self.disp_size): # Q updated according to disp_size
                
                print ("Q updated according to disp_size = " + str(self.disp_size))
                
                #dump Q to numpy
                self.predict_array = np.array(frameQ.queue)
                
                # call the predictor
                ##self.predict()
                print ("started prediction with " + str(self.disp_size) + " new elements in frame")
                
                i = 0
            
        
    def start(self):
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        #open socket
        sock.bind(('', self.port))
        
        print ("Socket opened")
        
        first_fill = True
        
        self.predict_array = []
        
        # init Q according to frame size
        frameQ = queue.Queue(maxsize=self.frame_size)
        
        self.run = True
        
        i = 0
        
        while self.run:
            measurement = []
            data, addr = sock.recvfrom(1024)
            print('received data', data)
            # split CSV
            split =  data.split(',')
            
            # fill measurement array
            if (len(split) > 9):
                measurement.append([float(split[2]), float(split[3]), float(split[4]), float(split[6]), float(split[7]), float(split[8])])    
            
            # fill Q initially
            if (first_fill):
                
                # fill Q
                frameQ.put_nowait(measurement)
                
                if (frameQ.full()):
                    
                    print ("Q initially filled")
                    
                    # dump Q to numpy array
                    self.predict_array = np.array(frameQ.queue)
                    
                    # call the predictor
                    ##self.predict()
                    print ("started prediction with initial Q")
                    
                    # next measurement will pop the first element of the Q
                    first_fill = False
            else:
                i = i + 1
                
                # pop first element of the Q
                frameQ.get()
                
                frameQ.put_nowait(measurement)
        
            if (i == self.disp_size): # Q updated according to disp_size
                
                print ("Q updated according to disp_size = " + str(self.disp_size))
                
                #dump Q to numpy
                self.predict_array = np.array(frameQ.queue)
                
                # call the predictor
                ##self.predict()
                print ("started prediction with " + str(self.disp_size) + " new elements in frame")
                
                i = 0
                
        def stop():
            self.run = False
            
detector = UDP_Detector(8888, 100, 10, True)
#detector.startWithPrerecordedData('')