# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:30:29 2015

@author: Ishay Tubi
"""
import os
import cv2
import numpy as np
import sys
import csv

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils import data

def mse_normlized(groundTruth, pred):
    delX = groundTruth[0]-groundTruth[2] 
    delY = groundTruth[1]-groundTruth[3] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance 
    diff = (pred-groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 

class RetVal:
    pass  ## A generic class to return multiple values without a need for a dictionary.


def createDataRowsFromCSV(csvFilePath, csvParseFunc, DATA_PATH, limit = sys.maxsize):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0    
    with open(csvFilePath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            d = csvParseFunc(row, DATA_PATH) 
            if d is not None and d.image is not None:
                data.append(d)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
                if len(data) % 100 == 0:
                    print(len(data))                
    return data

def createDataRowsFromGTB(csvFilePath, csvParseFunc, DATA_PATH, limit = sys.maxsize):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0    
    with open(csvFilePath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            d = csvParseFunc(row, DATA_PATH) 
            if d is not None and d.image is not None:
                data.append(d)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
                if len(data) % 100 == 0:
                    print(len(data))                
    return data

def getValidWithBBox(dataRows):
    import dlib
    R=RetVal()   
    R.outsideLandmarks = 0 
    R.noImages = 0 
    R.noFacesAtAll = 0 
    R.couldNotMatch = 0
    detector=dlib.get_frontal_face_detector()
    validRow=[]
    for dataRow in dataRows:
        if dataRow.image is None or len(dataRow.image)==0:
            R.noImages += 1
        lmd_xy = dataRow.landmarks().reshape([-1,2])
        left,  top = lmd_xy.min( axis=0 )
        right, bot = lmd_xy.max( axis=0 )
        det_bbox = None  # the valid bbox if found 
        dets = []
        if dataRow.image is not None and len(dataRow.image)!=0:
            dets = detector(np.array(dataRow.image, dtype = 'uint8'))
            for det in dets:
                det_box = BBox.BBoxFromLTRB(det.left(), det.top(), det.right(), det.bottom())

                # Does all landmarks fit into this box?
                if top >= det_box.top and bot<= det_box.bottom and left>=det_box.left and right<=det_box.right:
                    det_bbox = det_box                     
        if det_bbox is None:
            if len(dets)>0:
                R.couldNotMatch += 1  # For statistics, dlib found faces but they did not match our landmarks.
            else:
                R.noFacesAtAll += 1  # dlib found 0 faces.
        else:
            dataRow.fbbox = det_bbox  # Save the bbox to the data row
            if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[0] or det_bbox.bottom>dataRow.image.shape[1]:
                R.outsideLandmarks += 1  # Saftey check, make sure nothing goes out of bound.
            else:
                validRow.append(dataRow)  
                if (len(validRow)%100 == 0):
                    print(len(validRow))   
    return validRow,R 

def writeHD5(dataRows, outputPath, setTxtFilePATH, meanTrainSet, stdTrainSet , IMAGE_SIZE=40, mirror=False):
    ''' Create HD5 data set for caffe from given valid data rows.
    if mirror is True, duplicate data by mirroring. mirror
    ''' 
    from numpy import zeros
    import h5py   
    if mirror:
        BATCH_SIZE = len(dataRows) * 2
    else:
        BATCH_SIZE = len(dataRows) 
    HD5Images = zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
    HD5Landmarks = zeros([BATCH_SIZE, 10], dtype='float32')
    #prefix  = os.path.join(ROOT, 'caffeData', 'hd5', 'train')
    setTxtFile = open(setTxtFilePATH, 'w')       
    i = 0    
    for dataRowOrig in dataRows:
        if i % 1000 == 0 or i >= BATCH_SIZE-1:
            print ("Processing row %d " % (i+1)) 
            
        if not hasattr(dataRowOrig, 'fbbox'):  
            print ("Warning, no fbbox")
            continue       
        dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox)  # Get a cropped scale copy of the data row 
        scaledLM = dataRow.landmarksScaledMinus05_plus05() 
        image = dataRow.image.astype('f4')
        image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)  # normalize        
        HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
        #HD5Images[i, :] = image
        HD5Landmarks[i,:] = scaledLM
        i+=1
        # mirror--
        if mirror:
            dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox).copyMirrored()  # Get a cropped scale copy of the data row
            scaledLM = dataRow.landmarksScaledMinus05_plus05() 
            image = dataRow.image.astype('f4')
            image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)           
            
            HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            #HD5Images[i, :] = image
            HD5Landmarks[i,:] = scaledLM
            i+=1       
    with h5py.File(outputPath, 'w') as T:  
        T.create_dataset("X", data=HD5Images)  
        T.create_dataset("landmarks", data=HD5Landmarks)  
    setTxtFile.write(outputPath+"\n")
    setTxtFile.flush()
    setTxtFile.close()
    
def datarowToNPArray(dataRows, outputPath, setTxtFilePATH, meanTrainSet, stdTrainSet , IMAGE_SIZE=40, mirror=False):
    ''' Create HD5 data set for caffe from given valid data rows.
    if mirror is True, duplicate data by mirroring. 
    ''' 
    from numpy import zeros
    import h5py    
    if mirror:
        BATCH_SIZE = len(dataRows) *2
    else:
        BATCH_SIZE = len(dataRows) 
    HD5Images = zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], dtype='float32')
    HD5Landmarks = zeros([BATCH_SIZE, 10], dtype='float32')
    HD5Attributes = zeros([BATCH_SIZE, 4], dtype='float32')
    RawImages = []
    RawLandmarks = zeros([BATCH_SIZE, 10], dtype='float32')       
    i = 0    
    for dataRowOrig in dataRows:
        if i % 1000 == 0 or i >= BATCH_SIZE-1:
            print ("Processing row %d " % (i+1)) 
            
        if not hasattr(dataRowOrig, 'fbbox'):
            print ("Warning, no fbbox")
            continue       
        dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox)  # Get a cropped scale copy of the data row
        scaledLM = dataRow.landmarksScaledMinus05_plus05() 
        image = dataRow.image.astype('f4')
        image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
        
        #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
        HD5Images[i, :] = image
        HD5Landmarks[i,:] = scaledLM
        HD5Attributes[i,:] = dataRowOrig.gender, dataRowOrig.smiling, dataRowOrig.wearing_glasses, dataRowOrig.head_pose
        RawImages.append(dataRowOrig.path)
        RawLandmarks[i, :] = dataRowOrig.landmarks()
        i+=1
        
        if mirror:
            dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox).copyMirrored()  # Get a cropped scale copy of the data row
            scaledLM = dataRow.landmarksScaledMinus05_plus05() 
            image = dataRow.image.astype('f4')
            image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
            
            #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Images[i, :] = image
            HD5Landmarks[i,:] = scaledLM
            HD5Attributes[i,:] = dataRowOrig.gender, dataRowOrig.smiling, dataRowOrig.wearing_glasses, dataRowOrig.head_pose
            i+=1                
    return HD5Images, HD5Landmarks, HD5Attributes, RawImages, RawLandmarks

class ErrorAcumCUM:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(5, dtype ='f4') 
        self.itemsCounter = 0 #
        self.failureCounter1 = 0
        self.failureCounter2 = 0 
        self.failureCounter3 = 0 
        self.failureCounter4 = 0 
        self.failureCounter5 = 0 
        self.failureCounter6 = 0 
        self.failureCounter7 = 0 
        self.failureCounter8 = 0 
        self.failureCounter9 = 0 
        self.failureCounter10 = 0      
    def __repr__(self):
        return 'mean error:%s%%  %ditems' % (self.meanError(), self.itemsCounter)               
    def add(self, groundTruth, pred, index):
        normlized = mse_normlized(groundTruth, pred)
        self.errorPerLandmark += normlized   
        self.itemsCounter +=1
        if normlized.mean() < 0.05:
            self.failureCounter1 +=1
        if normlized.mean() <0.1 : 
            self.failureCounter2 +=1
        if normlized.mean() <0.15 : 
            self.failureCounter3 +=1
        if normlized.mean() <0.2 : 
            self.failureCounter4 +=1
        if normlized.mean() <0.25 : 
            self.failureCounter5 +=1
        if normlized.mean() <0.3 : 
            self.failureCounter6 +=1
        if normlized.mean() <0.35 : 
            self.failureCounter7 +=1
        if normlized.mean() <0.4 : 
            self.failureCounter8 +=1
        if normlized.mean() <0.45 : 
            self.failureCounter9 +=1
        if normlized.mean() <0.5 : 
            self.failureCounter10 +=1
    def meanError(self):
        if self.itemsCounter > 0:
            return [(self.errorPerLandmark/self.itemsCounter).mean()*100\
            , self.failureCounter1/self.itemsCounter*100\
            , self.failureCounter2/self.itemsCounter*100\
            , self.failureCounter3/self.itemsCounter*100\
            , self.failureCounter4/self.itemsCounter*100\
            , self.failureCounter5/self.itemsCounter*100\
            , self.failureCounter6/self.itemsCounter*100\
            , self.failureCounter7/self.itemsCounter*100\
            , self.failureCounter8/self.itemsCounter*100\
            , self.failureCounter9/self.itemsCounter*100\
            , self.failureCounter10/self.itemsCounter*100]
        else:
            return self.errorPerLandmark
    def plot(self):
        from matplotlib.pylab import show, plot, stem
        pass

class ErrorAcum:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(5, dtype ='f4') 
        self.itemsCounter = 0 
        self.failureCounter = 0 
        self.failureList = [] 
        self.suclist = []      
    def __repr__(self):
        return 'mean error:%f%% %s %d items,%d failures(>0.1) accuracy:%f%%' % (self.meanError().mean()*100, (self.errorPerLandmark / self.itemsCounter) if self.itemsCounter>0 else 0, self.itemsCounter, self.failureCounter, ((1-float(self.failureCounter)/self.itemsCounter) *100) if self.itemsCounter>0 else 0)               
    def add(self, groundTruth, pred, index):
        normlized = mse_normlized(groundTruth, pred)
        self.errorPerLandmark += normlized   
        self.itemsCounter +=1
        if normlized.mean() < 0.05:
            self.suclist.append(index)
        if normlized.mean() > 0.1: 
            # Count error above 10% as failure 
            self.failureCounter +=1
            self.failureList.append(index)
    def meanError(self):
        if self.itemsCounter > 0:
            return self.errorPerLandmark/self.itemsCounter
        else:
            return self.errorPerLandmark
    def printMeanError(self):
        if self.itemsCounter > 0:
            #print(self.errorPerLandmark / self.itemsCounter)
            strPrint = 'error above %%10:%f' % (1.0-float(self.failureCounter)/self.itemsCounter)
            print(strPrint)
            print('index of below 5%')
            print(self.suclist)
            print('index of lager than 10%')
            print(self.failureList)
    def __add__(self, x):
        ret = ErrorAcum()
        ret.errorPerLandmark = self.errorPerLandmark + x.errorPerLandmark
        ret.itemsCounter    = self.itemsCounter + x.itemsCounter
        ret.failureCounter  = self.failureCounter + x.failureCounter        
        return ret       
    def plot(self):
        from matplotlib.pylab import show, plot, stem
        pass

class BBox:  # Bounding box
    
    @staticmethod
    def BBoxFromLTRB(l, t, r, b):
        return BBox(l, t, r, b)   
    @staticmethod
    def BBoxFromXYWH_array(xywh):
        return BBox(xywh[0], xywh[1], +xywh[0]+xywh[2], xywh[1]+xywh[3])   
    @staticmethod
    def BBoxFromXYWH(x,y,w,h):
        return BBox(x,y, x+w, y+h)    
    def top_left(self):
        return (self.top, self.left)   
    def left_top(self):
        return (self.left, self.top)
    def bottom_right(self):
        return (self.bottom, self.right)   
    def right_top(self):
        return (self.right, self.top)  
    def relaxed(self, clip ,relax=3):  #@Unused
        from numpy import array
        _A = array
        maxWidth, maxHeight =  clip[0], clip[1]       
        nw, nh = self.size()*(1+relax)*.5       
        center = self.center()
        offset=_A([nw,nh])
        lefttop = center - offset
        rightbot= center + offset         
        self.left, self.top  = int( max( 0, lefttop[0] ) ), int( max( 0, lefttop[1]) )
        self.right, self.bottom = int( min( rightbot[0], maxWidth ) ), int( min( rightbot[1], maxHeight ) )
        return self
    def clip(self, maxRight, maxBottom):
        self.left = max(self.left, 0)
        self.top = max(self.top, 0)
        self.right = min(self.right, maxRight)
        self.bottom = min(self.bottom, maxBottom)       
    def size(self):
        from numpy import  array
        return array([self.width(), self.height()])    
    def center(self):
        from numpy import  array
        return array([(self.left+self.right)/2, (self.top+self.bottom)/2])               
    def __init__(self,left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom       
    def width(self):
        return self.right - self.left      
    def height(self):
        return self.bottom - self.top       
    def xywh(self):
        return self.left, self.top, self.width(), self.height()       
    def offset(self, x, y):
        self.left += x 
        self.right += x
        self.top += y 
        self.bottom += y        
    def scale(self, rx, ry):
        self.left *= rx 
        self.right *= rx
        self.top *= ry 
        self.bottom *= ry                       
    def __repr__(self):
        return 'left(%.1f), top(%.1f), right(%.1f), bottom(%.1f) w(%d) h(%d)' % (self.left, self.top, self.right, self.bottom,self.width(), self.height())
    def makeInt(self):
        self.left    = int(self.left)
        self.top     = int(self.top)
        self.right   = int(self.right)
        self.bottom  = int(self.bottom)
        return self

class DataRow:
    global TrainSetMean
    global TrainSetSTD
    IMAGE_SIZE = 40
    def __init__(self, path='', leftEye=(0, 0, ), rightEye=(0, 0), middle=(0, 0), leftMouth=(0, 0), rightMouth=(0, 0)):
        self.image = cv2.imread(path)
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.leftMouth = leftMouth
        self.rightMouth = rightMouth
        self.middle = middle
        self.name = os.path.split(path)[-1]
        self.sx = 1.
        self.sy = 1.
        self.offsetX = 0.
        self.offsetY = 0.
        #gender, smiling, wearing glasses, and head pose.
        self.gender = 1.
        self.smiling = 1.
        self.wearing_glasses = 1.
        self.head_pose = 1.
    def __repr__(self):
        return '{} le:{},{} re:{},{} nose:{},{}, lm:{},{} rm:{},{}'.format(
            self.name,
            self.leftEye[0], self.leftEye[1],
            self.rightEye[0], self.rightEye[1],
            self.middle[0], self.middle[1],
            self.leftMouth[0], self.leftMouth[1],
            self.rightMouth[0], self.rightMouth[1]
            )
    def setLandmarks(self,landMarks):
        """
        @landMarks : np.array
        set the landmarks from array
        """
        self.leftEye = landMarks[0:2]
        self.rightEye = landMarks[2:4]
        self.middle = landMarks[4:6]
        self.leftMouth = landMarks[6:8]
        self.rightMouth = landMarks[8:10]        
    
    def landmarks(self):
        # return numpy float array with ordered values
        stright = [
            self.leftEye[0],
            self.leftEye[1],
            self.rightEye[0],
            self.rightEye[1],
            self.middle[0],
            self.middle[1],
            self.leftMouth[0],
            self.leftMouth[1],
            self.rightMouth[0],
            self.rightMouth[1]]
        return np.array(stright, dtype='f4')

    
    def landmarksScaledMinus05_plus05(self):
        # return numpy float array with ordered values
        return self.landmarks().astype('f4')/40. - 0.5

    # resize landmars
    def scale(self, sx, sy):
        self.sx *= sx
        self.sy *= sy
        self.leftEye = (self.leftEye[0]*sx, self.leftEye[1]*sy)
        self.rightEye = (self.rightEye[0]*sx, self.rightEye[1]*sy)
        self.middle = (self.middle[0]*sx, self.middle[1]*sy)
        self.leftMouth = (self.leftMouth[0]*sx, self.leftMouth[1]*sy)
        self.rightMouth = (self.rightMouth[0]*sx, self.rightMouth[1]*sy)       
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1, 2)*[sx, sy]
        return self

    
    def offsetCropped(self, offset=(0., 0.)):
        """ given the cropped values - offset the positions by offset
        """
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1,2)-offset
        self.leftEye = (self.leftEye[0]-offset[0], self.leftEye[1]-offset[1])
        self.rightEye = (self.rightEye[0]-offset[0], self.rightEye[1]-offset[1])
        self.middle = (self.middle[0]-offset[0], self.middle[1]-offset[1])
        self.leftMouth = (self.leftMouth[0]-offset[0], self.leftMouth[1]-offset[1])
        self.rightMouth = (self.rightMouth[0]-offset[0], self.rightMouth[1]-offset[1])
        return self

    
    def inverseScaleAndOffset(self, landmarks):
        """ computes the inverse scale and offset of input data according to the inverse scale factor and inverse offset factor
        """
        from numpy import array; _A = array ; # Shothand 
        
        ret = _A(landmarks.reshape(-1,2)) *_A([1./self.sx, 1./self.sy])
        ret += _A([-self.offsetX, -self.offsetY])
        return ret

    @staticmethod
    def DataRowFromNameBoxInterlaved(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        d.leftEye = (float(row[5]), float(row[6]))
        d.rightEye = (float(row[7]), float(row[8]))
        d.middle = (float(row[9]), float(row[10]))
        d.leftMouth = (float(row[11]), float(row[12]))
        d.rightMouth = (float(row[13]), float(row[14]))
        return d
    
    @staticmethod
    def DataRowFromNameBoxInterlaved1(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        data = []  # the array we build
        validObjectsCounter = 0    
        with open(row[5], 'r') as landmarks_19:
            reader = csv.reader(landmarks_19, delimiter=' ')
            for row in reader:
                if(int(row[0]) == 8):
                    d.leftEye = (float(row[1]), float(row[2]))
                    continue
                elif(int(row[0]) == 11):
                    d.rightEye = (float(row[1]), float(row[2]))
                    continue
                elif(int(row[0]) == 14):
                    d.middle = (float(row[1]), float(row[2]))
                    continue
                elif(int(row[0]) == 16):
                    d.leftMouth = (float(row[1]), float(row[2]))
                    continue
                elif(int(row[0]) == 18):
                    d.rightMouth = (float(row[1]), float(row[2]))
                    continue
                else:
                    continue
        return d

    @staticmethod
    def DataRowFromMTFL(row, root=''):
        '''
        --x1...x5,y1...y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
        '''
        d = DataRow()
        if len(row[0]) <= 1:
            # bug in the files, it has spaces seperating them, skip it
            row=row[1:]            
        if len(row)<10:
            print ('error parsing ', row)
            return None
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        
        if d.image is None:
            print ('Error reading image', d.path)
            return None       
        d.leftEye = (float(row[1]), float(row[6]))
        d.rightEye = (float(row[2]), float(row[7]))
        d.middle = (float(row[3]), float(row[8]))
        d.leftMouth = (float(row[4]), float(row[9]))
        d.rightMouth = (float(row[5]), float(row[10]))
        d.gender = float(row[11])
        d.smiling = float(row[12])
        d.wearing_glasses = float(row[13])
        d.head_pose = float(row[14])
        return d

    @staticmethod
    def DataRowFromAFW(anno, root=''): # Assume data comming from parsed anno-v7.mat file.
        name = str(anno[0][0])
        # bbox = anno[1][0][0]
        # yaw, pitch, roll = anno[2][0][0][0]
        lm = anno[3][0][0]  # 6 landmarks
        if np.isnan(lm).any():
            return None  # Fail
        d = DataRow()
        d.path = os.path.join(root, name).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        #print(type(d.image))
        if d.image is None:
            print ('Error reading image', d.path)
            return None
        d.leftEye = (float(lm[0][0]), float(lm[0][1]))
        d.rightEye = (float(lm[1][0]), float(lm[1][1]))
        d.middle = (float(lm[2][0]), float(lm[2][1]))
        d.leftMouth = (float(lm[3][0]), float(lm[3][1]))
        # skip point 4 middle mouth - We take 0 left eye, 1 right eye, 2 nose, 3 left mouth, 5 right mouth
        d.rightMouth = (float(lm[5][0]), float(lm[5][1]))
        return d

    @staticmethod
    def DataRowFromPrediction(p, path='', image=None):
        d = DataRow(path)        
        p = (p+0.5)*40.  # scale from -0.5..+0.5 to 0..40
        
        d.leftEye = (p[0], p[1])
        d.rightEye = (p[2], p[3])
        d.middle = (p[4], p[5])
        d.leftMouth = (p[6], p[7])
        d.rightMouth = (p[8], p[9])
        return d

    def drawLandmarks(self, r=2, color=255, other=None, title=None):
        M = self.image
        if hasattr(self, 'prediction'):
            for x,y in self.prediction.reshape(-1,2):
                cv2.circle(M, (int(x), int(y)), r, (0,200,0), -1)            
        cv2.circle(M, (int(self.leftEye[0]), int(self.leftEye[1])), r, color, -1)
        cv2.circle(M, (int(self.rightEye[0]), int(self.rightEye[1])), r, color, -1)
        cv2.circle(M, (int(self.leftMouth[0]), int(self.leftMouth[1])), r, color, -1)
        cv2.circle(M, (int(self.rightMouth[0]), int(self.rightMouth[1])), r, color, -1)
        cv2.circle(M, (int(self.middle[0]), int(self.middle[1])), r, color, -1)
        if hasattr(self, 'fbbox'):
            cv2.rectangle(M, self.fbbox.top_left(), self.fbbox.bottom_right(), color)
        return M

    def show(self, r=2, color=255, other=None, title=None):
        M = self.drawLandmarks(r, color, other, title)
        if title is None:
            title = self.name
        cv2.imshow(title, M)
        return M

    def saveImage(self, filePath, r=2, color=255, other=None, title=None):
        M = self.drawLandmarks(r, color, other, title)
        if title is None:
            title = self.name
        # cv2.imshow(title, M)
        savePath = os.path.join(filePath,title)
        cv2.imwrite(savePath, M)
        # return M

    def makeInt(self):
        self.leftEye    = (int(self.leftEye[0]), int(self.leftEye[1]))
        self.rightEye   = (int(self.rightEye[0]), int(self.rightEye[1]))
        self.middle     = (int(self.middle[0]), int(self.middle[1]))
        self.leftMouth  = (int(self.leftMouth[0]), int(self.leftMouth[1]))
        self.rightMouth = (int(self.rightMouth[0]), int(self.rightMouth[1]))
        return self        

    def copyCroppedByBBox(self,fbbox, siz=np.array([40.,40.])):
        """
        @ fbbox : BBox
        Returns a copy with cropped, scaled to size
        """        
        fbbox.makeInt() # assume BBox class
        if fbbox.width()<10 or fbbox.height()<10:
            print ("Invalid bbox size:",fbbox)
            return None
            
        faceOnly = self.image[fbbox.top : fbbox.bottom, fbbox.left:fbbox.right, :]
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))        
        scaled.setLandmarks(self.landmarks())        
        """ @scaled: DataRow """
        scaled.offsetCropped(fbbox.left_top()) # offset the landmarks
        rx, ry = siz/faceOnly.shape[:2]
        scaled.scale(rx, ry)       
        return scaled        

    def copyMirrored(self):
        '''
        Return a copy with mirrored data (and mirrored landmarks).
        '''
        import numpy
        _A=numpy.array
        ret = DataRow() 
        ret.image=cv2.flip(self.image.copy(),1) 
        # Now we mirror the landmarks and swap left and right
        width = ret.image.shape[0]
        ret.leftEye = _A([width-self.rightEye[0], self.rightEye[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.rightEye = _A([width-self.leftEye[0], self.leftEye[1]])
        ret.middle = _A([width-self.middle[0], self.middle[1]])        
        ret.leftMouth = _A([width-self.rightMouth[0], self.rightMouth[1]]) # Toggle mouth positions and mirror x axis only
        ret.rightMouth = _A([width-self.leftMouth[0], self.leftMouth[1]])
        return ret


    def copyMirroredGenImage(self):
        '''
        Return a copy with mirrored data (and mirrored landmarks).
        '''
        import numpy
        _A=numpy.array
        ret = DataRow()
        ret.image=cv2.flip(self.image.copy(),1) 
        # Now we mirror the landmarks and swap left and right
        width = ret.image.shape[0]
        ret.leftEye = _A([width/40.0-self.rightEye[0]-1.0, self.rightEye[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.rightEye = _A([width/40.0-self.leftEye[0]-1.0, self.leftEye[1]])
        ret.middle = _A([width/40.0-self.middle[0]-1.0, self.middle[1]])
        ret.leftMouth = _A([width/40.0-self.rightMouth[0]-1.0, self.rightMouth[1]]) # Toggle mouth positions and mirror x axis only
        ret.rightMouth = _A([width/40.0-self.leftMouth[0]-1.0, self.leftMouth[1]])
        return ret

    @staticmethod
    def dummyDataRow():
        ''' Returns a dummy dataRow object to play with
        '''
        return DataRow('/Users/ishay/Dev/VanilaCNN/data/train/lfw_5590/Abbas_Kiarostami_0001.jpg',
                     leftEye=(106.75, 108.25),
                     rightEye=(143.75,108.75) ,
                     middle = (131.25, 127.25),
                     leftMouth = (106.25, 155.25),
                     rightMouth =(142.75,155.25)
                     )    

# pytorch data load
class DataPytorch(data.Dataset):
    def __init__(self,hd5_root,transforms=None):
        f = h5py.File(hd5_root,'r+')
        self.imgs = f['X']
        self.landmarks = f['landmarks']
        self.transforms = transforms
        # f.close()

    def __getitem__(self, index):
        img = self.imgs[index]
        array_img = np.asarray(img)
        my_data = T.from_numpy(array_img)
        landmark = self.landmarks[index]
        array_landmark = np.asarray(landmark)
        my_landmark = T.from_numpy(array_landmark)
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, my_landmark

    def __len__(self):
       return len(self.imgs)

class FcTcnnDataPytorchFCNet(data.Dataset):
    def __init__(self, hd5_root, labelsFile, index, transforms=None):
        f = h5py.File(hd5_root, 'r+')
        allOriginalImgs = f['X']
        allImages = f['P'] #featurePictures
        allLandmarks = f['landmarks']
        labels = np.loadtxt(labelsFile) 
        indexes = np.where(labels ==index)[0] 
        # print(indexes)
        while(len(indexes) == 0):
            index += 1
            indexes = np.where(labels ==index)[0]
        self.originalImgs = allOriginalImgs[list(indexes)]
        self.imgs = allImages[list(indexes)]
        self.landmarks = allLandmarks[list(indexes)]
        self.transforms = transforms       
        f.close()
    def __getitem__(self, index):
        img = self.imgs[index]
        my_data = T.from_numpy(img)
        landmark = self.landmarks[index]
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, landmark
    def __len__(self):
        return len(self.imgs)

class DataPytorchFCNet(data.Dataset):
    def __init__(self, hd5_root, labelsFile, index, transforms=None):
        f = h5py.File(hd5_root, 'r+')
        allOriginalImgs = f['X']
        allImages = f['F']
        allLandmarks = f['landmarks']
        labels = np.loadtxt(labelsFile) 
        indexes = np.where(labels ==index)[0] 
        # print(indexes)
        while(len(indexes) == 0):
            index += 1
            indexes = np.where(labels ==index)[0]
        self.originalImgs = allOriginalImgs[list(indexes)]
        self.imgs = allImages[list(indexes)]
        self.landmarks = allLandmarks[list(indexes)]
        self.transforms = transforms       
        f.close()
    def __getitem__(self, index):
        img = self.imgs[index]
        my_data = T.from_numpy(img)
        landmark = self.landmarks[index]
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, landmark
    def __len__(self):
        return len(self.imgs)

class DataPytorchFCNet1(data.Dataset):
    def __init__(self, hd5_root, labelsFile, index, transforms=None):
        f = h5py.File(hd5_root, 'r+')
        allOriginalImgs = f['X']
        allImages = f['F']
        allLandmarks = f['landmarks']
        labels = np.loadtxt(labelsFile) 
        indexes = np.where(labels ==index)[0] 
        # print(indexes)
        while(len(indexes) == 0):
            index += 1
            indexes = np.where(labels ==index)[0]
        self.originalImgs = allOriginalImgs[list(indexes)]
        self.imgs = allImages[list(indexes)]
        self.landmarks = allLandmarks[list(indexes)]
        self.transforms = transforms       
        f.close()
    def __getitem__(self, index):
        images = self.originalImgs[index]
        img = self.imgs[index]
        my_data = T.from_numpy(img)
        landmark = self.landmarks[index]
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, landmark, images
    def __len__(self):
        return len(self.imgs)


class FctcnnNet(nn.Module): 
    def __init__(self):
        super(FctcnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool1 = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 128*2*2
        self.conv5 = nn.Conv2d(64, 128, 2) #128*1*1
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(T.rrelu(self.conv1(x)))
        x = self.pool2(T.rrelu(self.conv2(x)))
        x = self.pool3(T.rrelu(self.conv3(x)))
        x = T.rrelu(self.conv4(x))
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x

class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool1 = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(T.tanh(self.conv1(x)))
        x = self.pool2(T.tanh(self.conv2(x)))
        x = self.pool3(T.tanh(self.conv3(x)))
        x = T.tanh(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = T.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class VanillaNetRReLU(nn.Module):
    def __init__(self):
        super(VanillaNetRReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool1 = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(T.rrelu(self.conv1(x)))
        x = self.pool2(T.rrelu(self.conv2(x)))
        x = self.pool3(T.rrelu(self.conv3(x)))
        x = T.rrelu(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = T.rrelu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyNetBase(nn.Module):
    def __init__(self):
        super(MyNetBase, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, 5)  # 5*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        # self.pool2 = nn.MaxPool2d(2, 2)  #48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        # self.pool3 = nn.MaxPool2d(2, 2)  #64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.pool(T.tanh(self.conv1(x)))
        x = self.pool(T.tanh(self.conv2(x)))
        x = self.pool(T.tanh(self.conv3(x)))
        x = T.tanh(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = T.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class MyNetBaseRReLU(nn.Module):
    def __init__(self):
        super(MyNetBaseRReLU, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, 5)  # 5*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        # self.pool2 = nn.MaxPool2d(2, 2)  #48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        # self.pool3 = nn.MaxPool2d(2, 2)  #64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.pool(F.rrelu(self.conv1(x)))
        x = self.pool(F.rrelu(self.conv2(x)))
        x = self.pool(F.rrelu(self.conv3(x)))
        x = F.rrelu(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = F.rrelu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyNetBase1(nn.Module):
    def __init__(self):
        super(MyNetBase1, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, 5)  # 5*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        # self.pool2 = nn.MaxPool2d(2, 2)  #48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        # self.pool3 = nn.MaxPool2d(2, 2)  #64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 6)
    def forward(self, x):
        x = self.pool(T.tanh(self.conv1(x)))
        x = self.pool(T.tanh(self.conv2(x)))
        x = self.pool(T.tanh(self.conv3(x)))
        x = T.tanh(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = T.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class FcTcnnFeatureExtractUseVanilla(nn.Module):
    def __init__(self):
        super(FcTcnnFeatureExtractUseVanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        #self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        #self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        # self.fc1 = nn.Linear(64 * 2 * 2, 100)
        # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(T.tanh(self.conv1(x)))
        x = self.pool(T.tanh(self.conv2(x)))
        x = self.pool(T.tanh(self.conv3(x)))
        x = T.tanh(self.conv4(x))
        y = x.view(-1, 64 * 2 * 2)
        # x = T.tanh(self.fc1(x))
        # x = self.fc2(x)
        return x, y

class FeatureExtractUseVanilla(nn.Module):
    def __init__(self):
        super(FeatureExtractUseVanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        #self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        #self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        # self.fc1 = nn.Linear(64 * 2 * 2, 100)
        # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(T.tanh(self.conv1(x)))
        x = self.pool(T.tanh(self.conv2(x)))
        x = self.pool(T.tanh(self.conv3(x)))
        x = T.tanh(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        # x = T.rrelu(self.fc1(x))
        # x = self.fc2(x)
        return x

class FeatureExtractUseVanillaRReLU(nn.Module):
    def __init__(self):
        super(FeatureExtractUseVanillaRReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # 3*40*40
        self.pool = nn.MaxPool2d(2, 2)  # 16*18*18
        self.conv2 = nn.Conv2d(16, 48, 3)
        #self.pool2 = nn.MaxPool2d(2, 2)  # 48*8*8
        self.conv3 = nn.Conv2d(48, 64, 3)
        #self.pool3 = nn.MaxPool2d(2, 2)  # 64*3*3
        self.conv4 = nn.Conv2d(64, 64, 2)  # 64*2*2
        # self.fc1 = nn.Linear(64 * 2 * 2, 100)
        # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(T.rrelu(self.conv1(x)))
        x = self.pool(T.rrelu(self.conv2(x)))
        x = self.pool(T.rrelu(self.conv3(x)))
        x = T.rrelu(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        # x = T.rrelu(self.fc1(x))
        # x = self.fc2(x)
        return x

class FcTcnnFullConnectedLayer(nn.Module): 
    def __init__(self):
        super(FcTcnnFullConnectedLayer, self).__init__()
        self.conv5 = nn.Conv2d(64,128,2)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv5(x)
        x = x.view(-1,128)
        x = self.fc1(x)
        return x

class FullConnectedLayer(nn.Module): 
    def __init__(self):
        super(FullConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class FullConnectedLayerRReLU(nn.Module): 
    def __init__(self):
        super(FullConnectedLayerRReLU, self).__init__()
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.rrelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ContextFCNet1(nn.Module):
    def __init__(self):
        super(ContextFCNet1, self).__init__()
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 8)
        self.fc3 = nn.Linear(64 * 2 * 2 + 8, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x1 = T.tanh(self.fc1(x))
        x1 = self.fc2(x1)
        x2 = T.cat((x, x1), 1)
        x2 = T.tanh(self.fc3(x2))
        x2 = self.fc4(x2)
        tmp = T.cat((x1,x2,), 1)
        res = T.zeros_like(tmp)
        res[:,0:4] = tmp[:,0:4]
        res[:,4:6] = tmp[:,8:10]
        res[:,6:10] = tmp[:,4:8]
        return res

class ContextFCNet2(nn.Module):
    def __init__(self):
        super(ContextFCNet2, self).__init__()
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 4)
        self.fc3 = nn.Linear(64 * 2 * 2 + 4, 100)
        self.fc4 = nn.Linear(100, 6)

    def forward(self, x):
        x1 = T.tanh(self.fc1(x))
        x1 = self.fc2(x1)
        x2 = T.cat((x, x1), 1)
        x2 = T.tanh(self.fc3(x2))
        x2 = self.fc4(x2)
        tmp = T.cat((x1, x2,), 1)
        return tmp

class ContextFCNet3(nn.Module):
    def __init__(self):
        super(ContextFCNet3, self).__init__()
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(64 * 2 * 2 + 4, 100)
        self.fc4 = nn.Linear(100, 6)

    def forward(self, x):
        x1 = T.tanh(self.fc1(x))
        x1 = self.fc2(x1)
        x2 = T.cat((x, x1[:,0:4]), 1)
        x2 = T.tanh(self.fc3(x2))
        x2 = self.fc4(x2)
        res = T.zeros_like(x1)
        res[:, 0:4] = x1[:, 0:4]
        res[:, 4:10] = x2
        return res

##########################
# load data of all clusters
class DataPytorchMyNet(data.Dataset):
    def __init__(self,encodingFeaturePath,numOfClusters,transforms=None):
        allImgs = np.zeros(shape=(0,5,40,40),dtype='float32')
        allLandmarks = np.zeros(shape=(0,10),dtype='float32')
        for i in range(numOfClusters):
            encodingFeatureName = 'encodingFeatureCluster%d.hd5' % (i)
            encodingHd5Path = os.path.join(encodingFeaturePath, encodingFeatureName)
            f0 = h5py.File(encodingHd5Path,'r+')
            imgs0 = f0['E']
            landmarks0 = f0['landmarks']
            allImgs = np.concatenate((allImgs,imgs0),axis=0) #https://blog.csdn.net/brucewong0516/article/details/79158758
            allLandmarks = np.concatenate((allLandmarks,landmarks0),axis=0)
            f0.close()
        self.imgs = allImgs
        self.landmarks = allLandmarks
        self.transforms = transforms
    def __getitem__(self, index):
        img = self.imgs[index]
        array = np.asarray(img)
        my_data = T.from_numpy(array)
        landmark = self.landmarks[index]
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, landmark
    def __len__(self):
       return len(self.imgs)

class DataPytorchMyNet1(data.Dataset):
    def __init__(self,encodingFeaturePath,numOfClusters,transforms=None):
        allOriginalImgs = np.zeros(shape=(0,3,40,40),dtype='float32')
        allImgs = np.zeros(shape=(0,5,40,40),dtype='float32')
        allLandmarks = np.zeros(shape=(0,10),dtype='float32')
        for i in range(numOfClusters):
            encodingFeatureName = 'encodingFeatureCluster%d.hd5' % (i)
            encodingHd5Path = os.path.join(encodingFeaturePath, encodingFeatureName)
            f0 = h5py.File(encodingHd5Path,'r+')
            OriginalImgs0 = f0['X']
            imgs0 = f0['E']
            landmarks0 = f0['landmarks']
            allOriginalImgs = np.concatenate((allOriginalImgs,OriginalImgs0),axis=0)
            allImgs = np.concatenate((allImgs,imgs0),axis=0) #https://blog.csdn.net/brucewong0516/article/details/79158758
            allLandmarks = np.concatenate((allLandmarks,landmarks0),axis=0)
            f0.close()
        self.imgs = allImgs
        self.landmarks = allLandmarks
        self.originalImgs = allOriginalImgs
        self.transforms = transforms
    def __getitem__(self, index):
        images = self.originalImgs[index]
        img = self.imgs[index]
        array = np.asarray(img)
        my_data = T.from_numpy(array)
        landmark = self.landmarks[index]
        if self.transforms:
            my_data = self.transforms(my_data)
        return my_data, landmark, images
    def __len__(self):
       return len(self.imgs)

# define my loss
class myNetLoss(nn.Module):
    def __init__(self):
        super(myNetLoss, self).__init__()
        return
    def forward(self, outputs, landmarks):
        batch_size = landmarks.shape[0]
        loss = 0
        for i in range(batch_size):
            a = T.norm(outputs[i,:]- landmarks[i,4:])
            b = T.norm(landmarks[i, 0:2]- landmarks[i,2:4])
            loss += T.div(a.pow(2), b.pow(2))
        return loss

class lossWithoutWeight(nn.Module): 
    def __init__(self):
        super(lossWithoutWeight,self).__init__()
        return
    def forward(self, outputs, landmarks):
        batch_size = landmarks.shape[0]
        loss = 0.0
        for i in range(batch_size):
            a = T.norm(outputs[i, :] - landmarks[i, :])
            b = T.norm(landmarks[i, 0:2] - landmarks[i, 2:4])
            loss += T.div(a.pow(2), b.pow(2)) 
        return loss

class lossWithWeight(nn.Module): 
    def __init__(self):
        super(lossWithWeight,self).__init__()
        return
    def forward(self, outputs, landmarks):
        batch_size = landmarks.shape[0]
        loss = 0.0
        for i in range(batch_size):
            a1 = T.norm(outputs[i,0:4] - landmarks[i,0:4])
            a2 = T.norm(outputs[i,4:] - landmarks[i,4:])
            b = T.norm(landmarks[i,0:2] - landmarks[i,2:4])
            loss += T.div(0.3*a1.pow(2)+a2.pow(2), b.pow(2))
        return loss

class Predictor:
    ROOT = '/home/haiduo/code'
    
    def preprocess(self, resized, landmarks):
        ret = resized.astype('f4')
        ret -= self.mean
        ret /= (1.e-6 + self.std)
        return ret, (landmarks / 40.) - 0.5
    
    def predict(self, resized):
        """
        @resized: image 40,40 already pre processed
        """
        self.net.blobs['data'].data[...] = cv2.split(resized)
        prediction = self.net.forward()['Dense2'][0]
        return prediction
    def __init__(self, protoTXTPath, weightsPath):
        import caffe
        caffe.set_mode_gpu()
        self.net = caffe.Net(protoTXTPath, weightsPath, caffe.TEST)
        self.mean = cv2.imread(os.path.join(Predictor.ROOT, 'trainMean.png')).astype('float')
        self.std = cv2.imread(os.path.join(Predictor.ROOT, 'trainSTD.png')).astype('float')

def calAccuracy(outputs, landmarks):
    batch_size = landmarks.shape[0]
    accuracy = T.zeros(5)
    for i in range(batch_size):
        b = T.norm(landmarks[i, 0:2] - landmarks[i, 2:4])
        a = T.randn(5)
        c = T.randn(5)
        for j in range(5):
            a[j] = T.norm(outputs[i, 2*j:2*j+2] - landmarks[i, 2*j:2*j+2])
            c[j] = a[j]/b
        accuracy += c
    return [accuracy[0].item(),accuracy[1].item(),accuracy[2].item(),accuracy[3].item(),accuracy[4].item()]

def calAccuracyMyNet(outputs, landmarks):
    batch_size = landmarks.shape[0]
    accuracy = T.zeros(3)
    for i in range(batch_size):
        b = T.norm(landmarks[i, 0:2] - landmarks[i, 2:4])
        a = T.randn(3)
        c = T.randn(3)
        for j in range(3):
            a[j] = T.norm(outputs[i, 2*j:2*j+2] -landmarks[i, 2*(j+2):2*(j+3)])
            c[j] = a[j]/b
        accuracy += c
    return [accuracy[0].item(),accuracy[1].item(),accuracy[2].item()]

# tensor 2 numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
