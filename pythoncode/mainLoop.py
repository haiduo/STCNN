import numpy as np
import os
import cv2
import sys
from pickle import load, dump
from zipfile import ZipFile
from urllib.request import urlretrieve  #python3

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

import h5py
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

from DataRow import DataRow, ErrorAcum, Predictor, createDataRowsFromCSV, getValidWithBBox, writeHD5,DataPytorch, VanillaNet,VanillaNetRReLU,FctcnnNet,\
    lossWithoutWeight, lossWithWeight ,myNetLoss, tensor_to_np, FeatureExtractUseVanilla,FeatureExtractUseVanillaRReLU, FcTcnnFullConnectedLayer, FullConnectedLayer, FullConnectedLayerRReLU, DataPytorchFCNet,DataPytorchFCNet1,\
     calAccuracy, ContextFCNet1, ContextFCNet2, ContextFCNet3,DataPytorchMyNet,DataPytorchMyNet1, MyNetBase,MyNetBaseRReLU,MyNetBase1, calAccuracyMyNet, mse_normlized, FcTcnnFeatureExtractUseVanilla, FcTcnnDataPytorchFCNet,\
     ErrorAcumCUM
 
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','~/caffe/distribute')  
sys.path.append(CAFFE_ROOT+'/python')  
# import caffe  # use pytorch instead of caffe
import dlib # Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
detector=dlib.get_frontal_face_detector() 

ROOT = '/home/haiduo/code'  
sys.path.append(os.path.join(ROOT, 'bishecode/pythoncode'))  

MTFL_TRAIN_SET = '/home/haiduo/code/data/MTFL/MTFL_trainSet.pickle' 
MTFL_TEST_SET = '/home/haiduo/code/data/MTFL/MTFL_testSet.pickle'

MEAN_TRAIN_SET = cv2.imread(os.path.join(ROOT, 'trainMean.png')).astype('f4') 
STD_TRAIN_SET  = cv2.imread(os.path.join(ROOT, 'trainSTD.png')).astype('f4')

AFW_DATA_PATH = os.path.join(ROOT, 'data', 'AFW/testimages') 
AFW_MAT_PATH = os.path.join(ROOT, 'data', 'AFW/anno-v7.mat')
AFW_TEST_PICKLE = os.path.join(ROOT, 'data', 'AFW/AFWTestSet.pickle') 

DATA_PATH = os.path.join(ROOT, 'data', 'LfwAndNet') 
CSV_TRAIN = os.path.join(ROOT, 'data', 'LfwAndNet/trainImageList.txt')
CSV_TEST  = os.path.join(ROOT, 'data', 'LfwAndNet/testImageList.txt') 
HD5_TRAIN_PATH = os.path.join(ROOT, 'data/caffeData/hd5','train.hd5') 
HD5_TEST_PATH = os.path.join(ROOT, 'data/caffeData/hd5','test.hd5')
CSV_GEN_TRAIN_PATH = os.path.join(ROOT, 'data', 'LfwAndNet/genTrain.txt') 
HD5_GEN_TRAIN_PATH = os.path.join(ROOT, 'data','dividedData/genTrain.hd5')

TRAIN_PATH = ROOT + '/data/dividedData/trainSet.hd5'
VALIDATION_PATH = ROOT + '/data/dividedData/validationSet.hd5'
TEST_PATH = ROOT + '/data/dividedData/testSet.hd5'

PATH_TO_WEIGHTS  = os.path.join(ROOT, 'bishecode/pythoncode/ZOO', 'vanillaCNN.caffemodel')  
PATH_TO_DEPLOY_TXT = os.path.join(ROOT, 'bishecode/pythoncode/ZOO', 'vanilla_deploy.prototxt')

#torch.cuda.curent_stream() 
torch.cuda.set_device(0) 
device = torch.device("cuda:0")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gmmPath = ROOT + '/model/gmmModel/LefNet/gmmModel.m' 
numOfCluster = 48  

###########################    STEPS TO RUN       ####################
''' 
'CreatetrainSetHD5', 'calcTrainSetMean'，'createAFW_TestSet'，'vanillaTestAFW'，'TCNNTestAFW'，'MyNetTestAFW'，
'createMTFL_TestSet'，'createMTFL_TrainSet'，'VanillaTestMTFL_AFLW'，'testMINIGen',
'TCNNTestMTFL_AFLW'，'MyNetTestMTFL_AFLW'，'genTrainSetHD5'，'checkGenData'，'vanillaTrainGenImage','dividGenDataSet'，'vanillaTrain'，'vanillaValidate'，
'vanillaTest'，'test_best_vanillaModel'，'getFeatureOfDividedData'，'GMMTrainValidateAndTest'，'getGenDataFeature'，'GMMGenData'，'TweakedTrainGen',
'getRightTrainValidateTestClusterLabels'，'getRightLabelsOfValidateAndTest'，'accuracyTestVanillaNet','accuracyTestVanillaNetGen',
'GMM'，'checkGMM'，'TweakedTrain'，'TweakedValidate','EncodingFeatureMap','trainMyNetBase','validateMyNetBase','getFeatureOfTestSet',
'GMMTestSet','checkGMMTestSet','accuracyTestTweakedNet','accuracytestMyNetBase','testMINI','getTestSetDataFeatureAndLabels','testMINITweakedGen'
'accuracyTestTweakedNetGen','EncodingFeatureMapGen', 'trainMyNetBaseGen', 'testMINIMyNetGen','accuracytestMyNetBaseGen','trainMyNetBaseGen_3'，
'testMINIMyNetGen_3','accuracytestMyNetBaseGen_3','testMINIVanillaGen','vanillaMINITestAFW','TCNNMINITestAFW','MyNetMINITestAFW','createAFLW_TestSet'
'getFeatureOfData', 'checkData'，‘testRangeOfLearningRate’
'''
STEPS = ['MyNetTestMTFL_AFLW']

################################################################
def getFeature(ind_model,dataSetPath,outHd5Path,batchsize): 
    model = FeatureExtractUseVanilla()
    model.cuda()
    name_model = 'model/modelVanillaNet/LefNetTrain/RReLU/VanillaEpoch%d.pkl' % (ind_model)
    MODEL_PATH_EPOCH = os.path.join(ROOT, name_model)
    save_model = torch.load(MODEL_PATH_EPOCH)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    # extract the feature
    dataset = DataPytorch(dataSetPath, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False)
    lengtth = len(dataset)
    # featureHD5 = np.zeros([lengtth,256])
    featureHD5 = np.zeros([lengtth, 256], dtype='float32')
    f = h5py.File(dataSetPath, 'r+')
    imgsHD5 = f['X']
    landmarksHD5 = f['landmarks']
    i = 0
    for data in dataloader:
        inputs, landmarks = data
        inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
        outputs = model(inputs)
        outputs_cpu = outputs.cpu()
        outputs_numpy = outputs_cpu.data.numpy()
        if (i+1)*batchsize>lengtth:
            featureHD5[i * batchsize:lengtth, :] = outputs_numpy
        else:
            featureHD5[i * batchsize:(i + 1) * batchsize, :] = outputs_numpy
        i += 1
    with h5py.File(outHd5Path, 'w') as T:  
        T.create_dataset("X", data=imgsHD5)  
        T.create_dataset("landmarks", data=landmarksHD5)  
        T.create_dataset("F", data = featureHD5) 

def getFcTcnnFeature(ind_model,dataSetPath,outHd5Path,batchsize): 
    model = FcTcnnFeatureExtractUseVanilla()
    model.cuda()
    name_model = 'modelFcTcnnNet/LefNetTrain/Tanh/FcTcnnEpoch%d.pkl' % (ind_model)
    MODEL_PATH_EPOCH = os.path.join(ROOT, 'model', name_model)
    save_model = torch.load(MODEL_PATH_EPOCH)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    # extract the feature
    dataset = DataPytorch(dataSetPath, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False)
    lengtth = len(dataset)
    # featureHD5 = np.zeros([lengtth,256])
    featurepicture = np.zeros([lengtth, 64, 2, 2], dtype='float32')
    featureHD5 = np.zeros([lengtth, 256], dtype='float32')
    f = h5py.File(dataSetPath, 'r+')
    imgsHD5 = f['X']
    landmarksHD5 = f['landmarks']
    i = 0
    for data in dataloader:
        inputs, landmarks = data
        inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
        imagepictures, outputs = model(inputs)
        images_cpu = imagepictures.cpu()
        images_numpy = images_cpu.data.numpy()
        outputs_cpu = outputs.cpu()
        outputs_numpy = outputs_cpu.data.numpy()
        if (i+1)*batchsize>lengtth:
            featureHD5[i * batchsize:lengtth, :] = outputs_numpy
            featurepicture[i * batchsize:lengtth, :] = images_numpy
        else:
            featureHD5[i * batchsize:(i + 1) * batchsize, :] = outputs_numpy
            featurepicture[i * batchsize:(i + 1) * batchsize, :] = images_numpy
        i += 1
    with h5py.File(outHd5Path, 'w') as T:  
        T.create_dataset("X", data=imgsHD5)  
        T.create_dataset("landmarks", data=landmarksHD5)  
        T.create_dataset("F", data = featureHD5) 
        T.create_dataset("P", data = featurepicture) 

def gmmModelTrain(featurePath, labelFile, gmmModelPath):
    f = h5py.File(featurePath, 'r+')
    dataMat = f['F']
    gmm = GaussianMixture(n_components=numOfCluster).fit(dataMat)
    joblib.dump(gmm, gmmModelPath)
    labels = gmm.predict(dataMat) 
    np.savetxt(labelFile, labels)
    print('GMMTrain success!')

def getLabelsByGmmModel(featurePath, labelFile, gmmModelPath): 
    f = h5py.File(featurePath, 'r+')
    dataMat = f['F']
    gmm = joblib.load(gmmModelPath)
    labels = gmm.predict(dataMat)
    np.savetxt(labelFile, labels)
    print('getLabelsByGmmModel success!')

################################################################

def prob(x,mu,sigma): 
    a = pow(np.e,float(-0.5 * pow((x-mu)/sigma, 2)))
    b = sigma * pow(2 * np.pi, 0.5)
    return a/b

def guassianKernel(everyCenters, kernelNum, mu, sigma, width, height, meanGray, stdGray):
    everyFeatureMap = np.zeros([kernelNum, width, height], dtype='float32')
    for i in range(kernelNum):
        center=everyCenters[i*2:(i+1)*2]
        tmp = center[0]
        center[0] = center[1]
        center[1] = tmp
        for m in range(width):
            for n in range(height):
                point = np.array([m,n])
                dist = np.linalg.norm(point-center)
                
                everyFeatureMap[i,m,n] = 255 * 4 * prob(dist,mu,sigma)
        everyFeatureMap[i,:,:] = (everyFeatureMap[i,:,:] - meanGray)/stdGray
    return everyFeatureMap

def encodingByRadialBasisKernel(centers, sigma, width, height):
    batchNum = len(centers)
    kernelNum = int(len(centers[0,:])/2)
    meanGray = cv2.cvtColor(MEAN_TRAIN_SET, cv2.COLOR_RGB2GRAY)
    stdGray = cv2.cvtColor(STD_TRAIN_SET, cv2.COLOR_RGB2GRAY)
    featureMap = np.zeros([batchNum, kernelNum, width, height], dtype='float32')
    for i in range(batchNum):
        featureMap[i,:,:,:] = guassianKernel(centers[i,:], kernelNum, 0, sigma, width, height, meanGray, stdGray)
    return featureMap


def getCentersImgsAndLandmarks(bestTweakedModel, featurePath, labelsPath, batch_size, sigma, width, height, encodingFeaturePath):
    for i in range(0,numOfCluster):
        print('cluster%d' % (i))
        FCNetName = 'FCNet'+str(i)
        locals()[FCNetName] = FullConnectedLayerRReLU() 
        
        locals()[FCNetName].cuda()
        # optimizerFCNet = 'optimizerFCNet' + str(i)
        # locals()[optimizerFCNet] = optim.Adam(locals()[FCNetName].parameters(), lr=0.0005)
        datasetName = 'dataset'+str(i)
        dataloaderName = 'dataloader'+str(i)
        locals()[datasetName] = DataPytorchFCNet(featurePath,labelsPath,i,transforms=None)
        locals()[dataloaderName] = DataLoader(locals()[datasetName], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        lengthCluster = len(locals()[datasetName])
        imgsCluster = locals()[datasetName].originalImgs
        landmarksCluster = locals()[datasetName].landmarks
        encodingFeatureMap = np.zeros([lengthCluster, 2 + 3, width, height], dtype='float32')
        encodingFeatureMap[:,0:3,:,:] = imgsCluster
        FCNetName = 'FCNet' + str(i)
        name_model = 'FCNet_epoch%d_Cluster%d.pkl' % (bestTweakedModel, i)
        FCNet_PATH_EPOCH = os.path.join(ROOT, 'model/modelFCNet/LfwNet/RReLU', name_model)
        # checkPoint = torch.load(FCNet_PATH_EPOCH)
        # locals()[FCNetName].load_state_dict(checkPoint['model'])
        locals()[FCNetName].load_state_dict(torch.load(FCNet_PATH_EPOCH))
        dataloaderName = 'dataloader' + str(i)
        countNum = 0
        for data in locals()[dataloaderName]:
            inputs, landmarks = data
            numAdd = len(inputs)
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = locals()[FCNetName](inputs)
            outputs = (outputs + 0.5) * 40
            outputs_cpu = outputs.cpu()
            outputs_numpy = outputs_cpu.data.numpy()
            encodingFeatureMap[batch_size*countNum:batch_size*countNum+numAdd,3:5,:,:] = encodingByRadialBasisKernel(outputs_numpy[:,0:4], sigma, width, height)
            countNum += 1
        encodingFeatureName = 'encodingFeatureCluster%d.hd5' % (i)
        encodingHd5Path = os.path.join(encodingFeaturePath, encodingFeatureName)
        with h5py.File(encodingHd5Path, 'w') as T: 
            T.create_dataset("X", data=imgsCluster)
            T.create_dataset("landmarks", data=landmarksCluster)
            T.create_dataset("E", data=encodingFeatureMap)

def getCentersImgsAndLandmarks1(bestTweakedModel, featurePath, labelsPath, batch_size, sigma, width, height, encodingFeaturePath):
    for i in range(0,numOfCluster):
        print('cluster%d' % (i))
        FCNetName = 'FCNet'+str(i)
        locals()[FCNetName] = FullConnectedLayer() 
        
        locals()[FCNetName].cuda()
        # optimizerFCNet = 'optimizerFCNet' + str(i)
        # locals()[optimizerFCNet] = optim.Adam(locals()[FCNetName].parameters(), lr=0.0005)
        datasetName = 'dataset'+str(i)
        dataloaderName = 'dataloader'+str(i)
        locals()[datasetName] = DataPytorchFCNet(featurePath,labelsPath,i,transforms=None)
        locals()[dataloaderName] = DataLoader(locals()[datasetName], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        lengthCluster = len(locals()[datasetName])
        #imgsCluster = locals()[datasetName].originalImgs
        landmarksCluster = locals()[datasetName].landmarks
        encodingFeatureMap = np.zeros([lengthCluster, 2, width, height], dtype='float32')
        #encodingFeatureMap[:,0:3,:,:] = imgsCluster
        FCNetName = 'FCNet' + str(i)
        name_model = 'FCNet_epoch%d_Cluster%d.pkl' % (bestTweakedModel, i)
        FCNet_PATH_EPOCH = os.path.join(ROOT, 'model/modelFCNet/LfwNet', name_model)
        # checkPoint = torch.load(FCNet_PATH_EPOCH)
        # locals()[FCNetName].load_state_dict(checkPoint['model'])
        locals()[FCNetName].load_state_dict(torch.load(FCNet_PATH_EPOCH))
        dataloaderName = 'dataloader' + str(i)
        countNum = 0
        for data in locals()[dataloaderName]:
            inputs, landmarks = data
            numAdd = len(inputs)
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = locals()[FCNetName](inputs)
            outputs = (outputs + 0.5) * 40
            outputs_cpu = outputs.cpu()
            outputs_numpy = outputs_cpu.data.numpy()
            encodingFeatureMap[batch_size*countNum:batch_size*countNum+numAdd,:,:,:] = encodingByRadialBasisKernel(outputs_numpy[:,0:4], sigma, width, height)
            countNum += 1
        encodingFeatureName = 'encodingFeatureCluster%d.hd5' % (i)
        encodingHd5Path = os.path.join(encodingFeaturePath, encodingFeatureName)
        with h5py.File(encodingHd5Path, 'w') as T: 
            #T.create_dataset("X", data=imgsCluster)
            T.create_dataset("landmarks", data=landmarksCluster)
            T.create_dataset("E", data=encodingFeatureMap)

########################################################

if 'calcTrainSetMean' in STEPS:
    dataRowsTrain_CSV = createDataRowsFromCSV(CSV_GEN_TRAIN_PATH, DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print("Finished reading %d rows from training data. Parsing BBox...." % len(dataRowsTrain_CSV))
    dataRowsTrainValid,R = getValidWithBBox(dataRowsTrain_CSV)
    print("Original train:",len(dataRowsTrain_CSV), "Valid Rows:", len(dataRowsTrainValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch)
    dataRowsTrain_CSV=[]  # remove from memory    
    print('Calculating train data mean value')
    meanTrainSet = np.zeros([40,40,3], dtype='double')
    for dataRow in dataRowsTrainValid:
        meanTrainSet += dataRow.copyCroppedByBBox(dataRow.fbbox).image.astype('double')    
    MEAN_TRAIN_SET = meanTrainSet / len(dataRowsTrainValid)   
    cv2.imwrite(os.path.join(ROOT, 'trainMean.png'), (MEAN_TRAIN_SET).astype('uint8'))  
    print('Finished Calculating train data mean value to file trainMean.png', MEAN_TRAIN_SET.mean())
    print('Calculating train data std value')
    stdTrainSet = np.zeros([40,40,3], dtype='double')
    for dataRow in dataRowsTrainValid:
        diff = dataRow.copyCroppedByBBox(dataRow.fbbox).image.astype('double') - MEAN_TRAIN_SET
        stdTrainSet += diff*diff        
    stdTrainSet /= len(dataRowsTrainValid)
    STD_TRAIN_SET = stdTrainSet**0.5   
    cv2.imwrite(os.path.join(ROOT, 'trainSTD.png'), (STD_TRAIN_SET).astype('uint8'))   
    print('Finished Calculating train data std value to file trainSTD.png with mean', STD_TRAIN_SET.mean())
else:
    MEAN_TRAIN_SET = cv2.imread(os.path.join(ROOT, 'trainMean.png')).astype('f4')
    STD_TRAIN_SET  = cv2.imread(os.path.join(ROOT, 'trainSTD.png')).astype('f4')

####################################################################
####################################################################
####################################################################
#############################################################
# divid genImage into trainSet 、validationSet (10000 77% ,2995 23%)

if 'CreateTrainSetHD5' in STEPS:
    dataRowsTrain_CSV = createDataRowsFromCSV(CSV_TRAIN, DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print("Finished reading %d rows from training data. Parsing BBox...." % len(dataRowsTrain_CSV))
    dataRowsTrainValid,R = getValidWithBBox(dataRowsTrain_CSV)
    print("Original train:",len(dataRowsTrain_CSV), "Valid Rows:", len(dataRowsTrainValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch)
    dataRowsTrain_CSV=[]  # remove from memory    
    writeHD5(dataRowsTrainValid, HD5_TRAIN_PATH, ROOT+'/caffeData/train.txt', MEAN_TRAIN_SET, STD_TRAIN_SET ,mirror=True)
    print("Finished writing train to caffeData/train.txt")

  
if 'CreateTestSetHD5' in STEPS:
    dataRowsTest_CSV = createDataRowsFromCSV(CSV_TEST, DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print("Finished reading %d rows from training data. Parsing BBox...." % len(dataRowsTest_CSV))
    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print("Original test:",len(dataRowsTest_CSV), "Valid Rows:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch)
    dataRowsTest_CSV = []  # remove from memory    
    writeHD5(dataRowsTestValid, HD5_TEST_PATH, ROOT+'/caffeData/test.txt', MEAN_TRAIN_SET, STD_TRAIN_SET ,mirror=True)
    print("Finished writing test to caffeData/test.txt")

if 'checkData' in STEPS: 
    f = h5py.File(HD5_TRAIN_PATH)
    imgs = f['X']
    for i in range(48):
        img0 = cv2.merge(imgs[i])
        img = img0 * (1.e-6 + STD_TRAIN_SET) + MEAN_TRAIN_SET
        imgName = '/home/haiduo/code/data/checkImages/img%d.jpg' % (i)
        cv2.imwrite(imgName, img)

################################################################################## 
batchsize = 8
from torch_lr_finder import LRFinder

if 'testRangeOfLearningRate' in STEPS:
    
    model = VanillaNet() 
    model = model.to(device=device, dtype=torch.double) 
    trainset = DataPytorch(HD5_TRAIN_PATH, transforms=transform)  
    trainloader = DataLoader(trainset, batch_size = batchsize, shuffle = True, num_workers = 0, drop_last = False, pin_memory = True)
    validset = DataPytorch(HD5_TEST_PATH, transforms=transform)  # genImage
    validloader = DataLoader(validset, batch_size=batchsize, shuffle = False, num_workers=0, drop_last=False,pin_memory=True)
    criterion = lossWithoutWeight
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader,val_loader=validloader ,end_lr=1, num_iter=100, step_mode="linear")
    lr_finder.plot(log_lr=False) # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

vanillaTrain_validLossPath = ROOT+'/lossAndAccuracy/LefNet/FcTcnnNetTrainLoss1.txt'
#######vanilla
if 'vanillaTrain' in STEPS:
    
    epochNum = 100 
    batchsize = 8 
    transform = None 
    net = VanillaNetRReLU() 
    net = net.to(device=device, dtype=torch.float) 
    
    # if torch.cuda.device_count() > 1:  
    #   net = nn.DataParallel(net)
    # net = net.to(device) 
    # net=nn.DataParallel(net,device_ids=[0,2])
    optimizer = optim.Adam(net.parameters(), lr=0.001) 
    print('Start training vanilla net...')
    trainset = DataPytorch(HD5_TRAIN_PATH, transforms=transform)  
    trainloader = DataLoader(trainset, batch_size = batchsize, shuffle = True, num_workers = 0, drop_last = False, pin_memory = True)
    validset = DataPytorch(HD5_TEST_PATH, transforms=transform)  
    validloader = DataLoader(validset, batch_size=batchsize, shuffle = False, num_workers=0, drop_last=False,pin_memory=True)
    print("size of trainset：", len(trainset), len(trainloader))
    print("size of validset：", len(validset), len(validloader))
    length_train = len(trainset)
    length_vaild = len(validset)
    for epoch in range(epochNum):
        if(epoch>=10 and epoch <20):
            optimizer.param_groups[0]["lr"] = 0.0005
        elif(epoch>=20 and epoch<50):
            optimizer.param_groups[0]["lr"] = 0.0001
        elif(epoch>=50 and epoch<100):
            optimizer.param_groups[0]["lr"] = 0.00001
        train_loss = 0.0
        trainallAccuracy = 0.0
        validallAccuracy = 0.0
        traineveryAccuracy = [0,0,0,0,0]
        valideveryAccuracy = [0,0,0,0,0]
        for i, data in enumerate(trainloader, 0):
            inputs, landmarks = data
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            optimizer.zero_grad() 
            #forward + backward + optimize
            outputs = net(inputs)
            loss = lossWithoutWeight(outputs, landmarks)
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
        
        ######
        name_model = 'FcTcnnEpoch%d.pkl' %(epoch+1)
        MODEL_PATH_EPOCH = os.path.join(ROOT, 'model/modelFcTcnnNet/LefNetTrain/Tanh', name_model)
        torch.save(net.state_dict(), MODEL_PATH_EPOCH) 
        for data1, data2 in zip(trainloader, validloader):
            ##########traindata
            inputs, landmarks = data1
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = net(inputs)
            accuracyList = calAccuracy(outputs, landmarks)      
            traineveryAccuracy = [m+n for m,n in zip(traineveryAccuracy,accuracyList)]
            trainallAccuracy += sum(accuracyList)
            ##########validdata
            inputs, landmarks = data2
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = net(inputs)
            accuracyList = calAccuracy(outputs, landmarks)
            valideveryAccuracy = [m+n for m,n in zip(valideveryAccuracy,accuracyList)]
            validallAccuracy += sum(accuracyList)
        trainallAccuracy = trainallAccuracy / (length_train * 5)
        validallAccuracy = validallAccuracy / (length_vaild * 5)
        loss_accuracy = open(vanillaTrain_validLossPath, 'a+')
        strPrintAll = 'epoch:%d trainloss:%0.3f trainAccuracy:%.5f validateAccuracy:%.5f trainEvery:%.5f %.5f %.5f %.5f %.5f validEvery:%.5f %.5f %.5f %.5f %.5f\n' % \
             (epoch+1, train_loss/length_train, trainallAccuracy, validallAccuracy, traineveryAccuracy[0]/length_train, traineveryAccuracy[1]/length_train, traineveryAccuracy[2]/length_train, traineveryAccuracy[3]/length_train, traineveryAccuracy[4]/length_train,\
             valideveryAccuracy[0]/length_vaild, valideveryAccuracy[1]/length_vaild, valideveryAccuracy[2]/length_vaild, valideveryAccuracy[3]/length_vaild, valideveryAccuracy[4]/length_vaild)
        print(strPrintAll)
        loss_accuracy.write(strPrintAll)
        loss_accuracy.close()       
    print('Finish Vanilla training!')

bestModleIndex = 34
transform = None
if 'testMINI' in STEPS:
    name_model = 'VanillaEpoch%d.pkl' % (bestModleIndex)
    MODEL_PATH_TEST = os.path.join(ROOT, 'model/modelVanillaNet/LefNetTrain', name_model)
    HD5_TESTMIN_PATH = HD5_TEST_PATH
    model = VanillaNet()
    model.cuda()
    model.load_state_dict(torch.load(MODEL_PATH_TEST))
    # load testSet
    dataset = DataPytorch(HD5_TESTMIN_PATH, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last= False)
    # get some images randomly
    dataiter = iter(dataloader)
    images, landmarks = dataiter.next()
    # use trained model predict landmarks
    images, landmarks = images.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
    outputs = model(images)
    landmarks = landmarks.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    landmarks = (landmarks+0.5)*40
    outputs = (outputs+0.5)*40
    batch_size = images.shape[0]
    for i in range(batch_size):
        img = images[i,:,:,:].cpu().detach().numpy().transpose((1,2,0))
        img = img*(1.e-6 + STD_TRAIN_SET)+MEAN_TRAIN_SET
        prediction_model = outputs[i,:].reshape(-1,2)
        truth_landmarks = landmarks[i,:].reshape(-1,2)
        for x, y in truth_landmarks:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)  # blue is ground true
        for x, y in prediction_model:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)  # green is predicted 
        title = "img%d.jpg" %(i)
        imgName = os.path.join(ROOT, 'data/TestMINIResults/Vanilla/LefNet', title)
        # cv2.imshow(title, img)
        cv2.imwrite(imgName,img)

##############################################################################GMM
trainFeaturePath = ROOT + '/data/featureAndLabelsOfGen/LfwNet/trainFeatureRReLU.hd5'
validateFeaturePath = ROOT + '/data/featureAndLabelsOfGen/LfwNet/validateFeatureRReLU.hd5'
# testFeaturePath = ROOT + '/data/featureAndLabelsOfGen/LfwNet/testFeature.hd5'
if 'getFeatureOfData' in STEPS:
    batchsize=8
    print('Start getFeatureOfDividedDate...')
    getFeature(bestModleIndex, HD5_TRAIN_PATH, trainFeaturePath, batchsize)
    print('Finish geting feaure of trainSet!')
    getFeature(bestModleIndex, HD5_TEST_PATH, validateFeaturePath, batchsize)
    print('Finish geting feature of validateSet!')
    # getFeature(bestModleIndex, TEST_PATH, testFeaturePath, batchsize)
    # print('Finish geting feature of testSet!')

trainLabelsPath = ROOT + '/data/featureAndLabelsOfGen/LfwNet/trainLabelsRReLU.txt'
validateLabelsPath = ROOT + '/data/featureAndLabelsOfGen/LfwNet/validateLabelsRReLU.txt'
gmmPath = '/home/haiduo/code/model/gmmModel/LefNet/gmmRReLU'
# testLabelsPath = ROOT + '/data/featureAndLabelsOfGen/testLabels.txt'
if 'GMMTrainValidateAndTest' in STEPS:
    print('Start gmm trainFeature...')
    gmmModelTrain(trainFeaturePath, trainLabelsPath, gmmPath) 
    print('Get labels of trainFeature successfully!')
    getLabelsByGmmModel(validateFeaturePath, validateLabelsPath, gmmPath)
    print('Get labels of validate Feature successfully!')
    # getLabelsByGmmModel(testFeaturePath, testLabelsPath , gmmPath)
    # print('Get labels of testFeature successfully!')

#################################################################KNN
def getCentriods(dataPath, labelFilePath):
    f = h5py.File(dataPath, 'r+')
    data = f['F']
    sizeOfFeature = len(data[0, :])
    labels = np.loadtxt(labelFilePath)
    centroidsData = np.zeros((numOfCluster, sizeOfFeature))
    for i in range(numOfCluster):
        indexes = np.where(labels == i)[0]
        numsOfImages = len(indexes)
        if(numsOfImages == 0):
            continue
        imgsOfEveryCluster = data[list(indexes)]
        for j in range(numsOfImages):
            centroidsData[i,:] += imgsOfEveryCluster[j,:]
        centroidsData[i,:] /= numsOfImages
    return centroidsData

def distCentroids(centroids1_train,centroids2):
    distMat = np.zeros((numOfCluster,numOfCluster))
    distMat = np.sqrt(-2*np.dot(centroids2, centroids1_train.T) + np.sum(np.square(centroids1_train), axis = 1) + np.transpose([np.sum(np.square(centroids2), axis = 1)]))
    correspondingIndex = np.argmin(distMat,axis=1) 
    return correspondingIndex

def getRightLabelsFile(Data1,labelsPath1,Data2,labelsPath2,rightLabelsPath):
    centroidsTrain = getCentriods(Data1, labelsPath1)
    centroidsValidate = getCentriods(Data2, labelsPath2)
    indesValidate = distCentroids(centroidsTrain, centroidsValidate)
    labelsValidate = np.loadtxt(labelsPath2)
    rightLabelsValidate = np.zeros_like(labelsValidate)
    # rightLabelsValidate = [indesValidate[k] for k in labelsValidate]
    for i in range(len(labelsValidate)):
        rightLabelsValidate[i] = indesValidate[int(labelsValidate[i])]
    np.savetxt(rightLabelsPath, rightLabelsValidate)

RightValidateLabelsPath = '/home/haiduo/code/data/featureAndLabelsOfGen/LfwNet/rightValidateLabelsRReLU.txt'
# RightTestLabelsPath = '/home/haiduo/code/data/featureAndLabelsOfGen/rightTestLabels.txt'
if 'getRightLabelsOfValidateAndTest' in STEPS:
    print('Start get Right Labels Of Validate And Test...')
    getRightLabelsFile(trainFeaturePath, trainLabelsPath, validateFeaturePath, validateLabelsPath, RightValidateLabelsPath)
    # getRightLabelsFile(trainFeaturePath, trainLabelsPath, testFeaturePath, testLabelsPath, RightTestLabelsPath )
    print('Finish get Right Labels Of Validate And Test!')

#########################################################################FCNet
epochNumFCNet = 100
TweakedTrain_validLossPath = ROOT+'/lossAndAccuracy/LefNet/TweakedTrainLossRReLU.txt'
FCNetLoss_EveryEpoch = ROOT+'/lossAndAccuracy/LefNet/TweakedtlossEveryClusterRReLU.txt'
bestModleIndex = 34  #34 or 36 or 38
batchsize = 8   
if 'TweakedTrain'in STEPS:
    print('Start TweakedTrain...')
    length = 0
    length_train = 0
    length_vaild = 0
    lengthCluster = np.zeros(numOfCluster)    
    
    for i in range(numOfCluster):
        FCNetName = 'FCNet'+str(i)
        locals()[FCNetName] = FullConnectedLayerRReLU() 
        locals()[FCNetName].cuda()
        name_model = 'model/modelVanillaNet/LefNetTrain/RReLU/VanillaEpoch%d.pkl' % (bestModleIndex) 
        MODEL_PATH_EPOCH = os.path.join(ROOT, name_model)
        save_model = torch.load(MODEL_PATH_EPOCH)
        model_dict = locals()[FCNetName].state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        locals()[FCNetName].load_state_dict(model_dict)
        optimizerFCNet = 'optimizerFCNet' + str(i)
        locals()[optimizerFCNet] = optim.Adam(locals()[FCNetName].parameters(), lr=0.0005)
        datasetName = 'dataset'+str(i)
        dataloaderName = 'dataloader'+str(i)
        locals()[datasetName] = DataPytorchFCNet(trainFeaturePath,trainLabelsPath,i,transforms=None) 
        locals()[dataloaderName] = DataLoader(locals()[datasetName], batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        lengthCluster[i] = len(locals()[datasetName])
        length += lengthCluster[i] #19718
    for epoch in range(epochNumFCNet):
        ######Training
        if(epoch>=10 and epoch<20):
            locals()[optimizer].param_groups[0]["lr"] = 0.0001
        elif(epoch>=20 and epoch<50):
            locals()[optimizer].param_groups[0]["lr"] = 0.00001
        elif(epoch>=50 and epoch<100):
            locals()[optimizer].param_groups[0]["lr"] = 0.000001
        running_loss_epoch = 0.0
        for i in range(numOfCluster):
            running_loss_cluster = 0
            FCNetName = 'FCNet' + str(i)
            optimizer = 'optimizerFCNet' + str(i)
            dataloaderName = 'dataloader' + str(i)
            for data in locals()[dataloaderName]:
                inputs, landmarks = data
                inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
                locals()[optimizer].zero_grad()
                outputs = locals()[FCNetName](inputs)
                loss = lossWithoutWeight(outputs, landmarks)
                loss.backward()
                locals()[optimizer].step()
                running_loss_cluster += loss.item() 
                running_loss_epoch += loss.item()      
                     
            name_model = '/modelFCNet/LfwNet/RReLU/FCNet_epoch%d_Cluster%d.pkl' % (epoch+1, i)
            FCNet_PATH_EPOCH = ('/home/haiduo/code/model' + name_model)
            torch.save(locals()[FCNetName].state_dict(), FCNet_PATH_EPOCH)
            mean_loss_cluster = running_loss_cluster/lengthCluster[i]
            file_loss_cluster = open(FCNetLoss_EveryEpoch, 'a+')
            strPrint = 'epoch%dClusterMeanLoos%d: %.5f\n' % (epoch+1, i, mean_loss_cluster)
            #print(strPrint)
            file_loss_cluster.write(strPrint)
            file_loss_cluster.close()   
            ####### set dataset
            traindatasetName = 'traindataset'+str(i)
            traindataloaderName = 'traindataloader'+str(i)
            validatedatasetName = 'validatedataset'+str(i)
            validatedataloaderName = 'validatedataloader'+str(i)
            locals()[traindatasetName] = DataPytorchFCNet(trainFeaturePath, trainLabelsPath, i, transforms=None) 
            locals()[traindataloaderName] = DataLoader(locals()[traindatasetName], batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
            length_train += len(locals()[traindatasetName])
            locals()[validatedatasetName] = DataPytorchFCNet(validateFeaturePath, RightValidateLabelsPath, i, transforms=None) 
            locals()[validatedataloaderName] = DataLoader(locals()[validatedatasetName], batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
            length_vaild += len(locals()[validatedatasetName])
            trainallAccuracy = 0
            validallAccuracy = 0
            traineveryAccuracy = [0,0,0,0,0]
            valideveryAccuracy = [0,0,0,0,0]
            ########### accuracy
            for data1, data2 in zip(locals()[traindataloaderName], locals()[validatedataloaderName]):
                ##########traindata
                inputs, landmarks = data1
                inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)             
                outputs = locals()[FCNetName](inputs)
                accuracyList = calAccuracy(outputs, landmarks)
                traineveryAccuracy =  [m+n for m,n in zip(traineveryAccuracy, accuracyList)]
                trainallAccuracy += sum(accuracyList)
                ##########validdata
                inputs, landmarks = data2
                inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
                outputs = locals()[FCNetName](inputs)
                accuracyList = calAccuracy(outputs, landmarks)
                valideveryAccuracy =  [m+n for m,n in zip(valideveryAccuracy, accuracyList)]
                validallAccuracy += sum(accuracyList)
        trainallAccuracy = trainallAccuracy / (length_train * 5)
        validallAccuracy = validallAccuracy / (length_vaild * 5)
        loss_accuracy = open(TweakedTrain_validLossPath, 'a+')
        strPrintAll = 'epoch:%d trainloss:%0.3f trainAccuracy:%.5f validateAccuracy:%.5f trainEvery:%.5f %.5f %.5f %.5f %.5f validEvery:%.5f %.5f %.5f %.5f %.5f\n' % \
            (epoch+1, running_loss_epoch/length, trainallAccuracy, validallAccuracy, traineveryAccuracy[0]/length_train, traineveryAccuracy[1]/length_train, traineveryAccuracy[2]/length_train, traineveryAccuracy[3]/length_train, traineveryAccuracy[4]/length_train,\
            valideveryAccuracy[0]/length_vaild, valideveryAccuracy[1]/length_vaild, valideveryAccuracy[2]/length_vaild, valideveryAccuracy[3]/length_vaild, valideveryAccuracy[4]/length_vaild)
        print(strPrintAll)
        loss_accuracy.write(strPrintAll)
        loss_accuracy.close() 
    print('Finish training TweakedNet!')

#################################################################################### 
encodingFeaturePathTrain = ROOT + '/data/encodingFeature/encodingFeatureTrain/LfwNet/40/'
encodingFeaturePathValidate = ROOT + '/data/encodingFeature/encodingFeatureValidate/LfwNet/40/'
#encodingFeaturePathTest = ROOT + '/data/encodingFeature/encodingFeatureTest/'
bestTCNNIndex = 3
if 'EncodingFeatureMap' in STEPS:
    print('Start encoding feature map...')
    getCentersImgsAndLandmarks(bestTCNNIndex, trainFeaturePath,trainLabelsPath, batchsize, 2, 40, 40,encodingFeaturePathTrain)
    getCentersImgsAndLandmarks(bestTCNNIndex, validateFeaturePath,RightValidateLabelsPath, batchsize, 2, 40, 40,encodingFeaturePathValidate)
    #getCentersImgsAndLandmarks(bestTCNNIndex, testFeaturePath,RightTestLabelsPath , 16, 2, 40, 40,encodingFeaturePathTest)
    print('Finish EncodingFeatureMap!')

encodingFeaturePathTrain18 = ROOT + '/data/encodingFeature/encodingFeatureTrain/LfwNet/18/'
encodingFeaturePathValidate18 = ROOT + '/data/encodingFeature/encodingFeatureValidate/LfwNet/18/'
if 'EncodingFeatureMap1' in STEPS:
    print('Start encoding feature map...')
    getCentersImgsAndLandmarks1(bestTCNNIndex, trainFeaturePath,trainLabelsPath, batchsize, 2, 40, 40,encodingFeaturePathTrain18)
    getCentersImgsAndLandmarks1(bestTCNNIndex, validateFeaturePath,RightValidateLabelsPath, batchsize, 2, 40, 40,encodingFeaturePathValidate18)
    print('Finish EncodingFeatureMap!')

####################################################################################### 
batchsize = 1
from torch_lr_finder import LRFinder

MyNetTrainLossPath = ROOT+'/lossAndAccuracy/LefNet/MyNetTrainLossRReLU.txt' 
bestTCNNIndex = 3
batchsize = 8
if 'trainMyNetBase' in STEPS:
    print('Start trainMyNetBase...')
    epochNum=100
    MyNet = MyNetBaseRReLU()
    MyNet = MyNet.to(device=device, dtype=torch.float)
    optimizer = optim.Adam(MyNet.parameters(), lr=0.001)
    trainset = DataPytorchMyNet(encodingFeaturePathTrain, 48, transforms=transform)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    validset = DataPytorchMyNet(encodingFeaturePathValidate, 48, transforms=transform)
    validloader = DataLoader(validset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    print("size of trainset：", len(trainset), len(trainloader))
    print("size of validset：", len(validset), len(validloader))
    length_train = len(trainset) 
    length_vaild = len(validset) 
    for epoch in range(0, epochNum):
        if (epoch >= 10 and epoch < 20):
            optimizer.param_groups[0]["lr"] = 0.0005
        elif (epoch >= 20 and epoch < 50):
            optimizer.param_groups[0]["lr"] = 0.0001
        elif (epoch >= 50 and epoch < 100):
            optimizer.param_groups[0]["lr"] = 0.00001
        train_loss = 0.0
        trainallAccuracy = 0.0
        validallAccuracy = 0.0
        traineveryAccuracy = [0,0,0,0,0]
        valideveryAccuracy = [0,0,0,0,0]
        for data in trainloader:
            inputs, landmarks = data
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = MyNet(inputs)
            #loss = myNetLoss(outputs, landmarks)
            loss = lossWithoutWeight(outputs, landmarks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        name_model = 'MyNet_epoch%d.pkl' % (epoch+1)
        MODEL_PATH_EPOCH = ('/home/haiduo/code/model/modelmyNet/LfwNet/RReLU/'+ name_model)
        torch.save(MyNet.state_dict(), MODEL_PATH_EPOCH)
        for data1, data2 in zip(trainloader, validloader):
            ##########traindata
            inputs, landmarks = data1
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = MyNet(inputs)
            accuracyList = calAccuracy(outputs, landmarks)  
            #accuracyList = calAccuracyMyNet(outputs, landmarks)      
            traineveryAccuracy = [m+n for m,n in zip(traineveryAccuracy,accuracyList)]
            trainallAccuracy += sum(accuracyList)
            ##########validdata
            inputs, landmarks = data2
            inputs, landmarks = inputs.to(device=device, dtype=torch.float), landmarks.to(device=device, dtype=torch.float)
            outputs = MyNet(inputs)
            accuracyList = calAccuracy(outputs, landmarks)
            # accuracyList = calAccuracyMyNet(outputs, landmarks)
            valideveryAccuracy = [m+n for m,n in zip(valideveryAccuracy,accuracyList)]
            validallAccuracy += sum(accuracyList)
        #trainallAccuracy = trainallAccuracy / (length_train * 3)
        #validallAccuracy = validallAccuracy / (length_vaild * 3)
        trainallAccuracy = trainallAccuracy / (length_train * 5)
        validallAccuracy = validallAccuracy / (length_vaild * 5)
        loss_accuracy = open(MyNetTrainLossPath, 'a+')
        # strPrintAll = 'epoch:%d trainloss:%0.3f trainAccuracy:%.5f validateAccuracy:%.5f trainEvery:%.5f %.5f %.5f validEvery:%.5f %.5f %.5f\n' % \
        #      (epoch+1, train_loss/length_train, trainallAccuracy, validallAccuracy, traineveryAccuracy[0]/length_train, traineveryAccuracy[1]/length_train, traineveryAccuracy[2]/length_train,\
        #      valideveryAccuracy[0]/length_vaild, valideveryAccuracy[1]/length_vaild, valideveryAccuracy[2]/length_vaild)
        strPrintAll = 'epoch:%d trainloss:%0.3f trainAccuracy:%.5f validateAccuracy:%.5f trainEvery:%.5f %.5f %.5f %.5f %.5f validEvery:%.5f %.5f %.5f %.5f %.5f\n' % \
             (epoch+1, train_loss/length_train, trainallAccuracy, validallAccuracy, traineveryAccuracy[0]/length_train, traineveryAccuracy[1]/length_train, traineveryAccuracy[2]/length_train, traineveryAccuracy[3]/length_train, traineveryAccuracy[4]/length_train,\
             valideveryAccuracy[0]/length_vaild, valideveryAccuracy[1]/length_vaild, valideveryAccuracy[2]/length_vaild, valideveryAccuracy[3]/length_vaild, valideveryAccuracy[4]/length_vaild)
        print(strPrintAll)
        loss_accuracy.write(strPrintAll)
        loss_accuracy.close()   
    print('Finish MyNet training!')

######################################################## 
def preprocess(resized, landmarks):
    ret = resized.astype('f4')
    ret -= MEAN_TRAIN_SET
    ret /= (1.e-6 + STD_TRAIN_SET)
    return ret, (landmarks / 40.) - 0.5

def loadFeatureExtractNet(ind_model): 
    model = FeatureExtractUseVanillaRReLU() 
    model.cuda()
    name_model = 'VanillaEpoch%d.pkl' % (ind_model)  
    MODEL_PATH_EPOCH = os.path.join(ROOT, 'model/modelVanillaNet/LefNetTrain/RReLU/', name_model) 
    save_model = torch.load(MODEL_PATH_EPOCH)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()} 
    model_dict.update(state_dict) 
    model.load_state_dict(model_dict)
    return model

def loadFeatureExtractNetTcFcnn(ind_model): 
    model = FcTcnnFeatureExtractUseVanilla()
    model.cuda()
    name_model = 'FcTcnnEpoch%d.pkl' % (ind_model)  
    MODEL_PATH_EPOCH = os.path.join(ROOT, 'model/modelFcTcnnNet/LefNetTrain/Tanh', name_model) 
    save_model = torch.load(MODEL_PATH_EPOCH)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()} 
    model_dict.update(state_dict) 
    model.load_state_dict(model_dict)
    return model

def loadFullNet(ind_model): 
    fullNet = []
    for i in range(numOfCluster):
        FCNetName = 'FCNet'+str(i)
        locals()[FCNetName] = FullConnectedLayerRReLU() 
        locals()[FCNetName].cuda()
        name_model = 'FCNet_epoch%d_Cluster%d.pkl' % (ind_model, i)
        FCNet_PATH_EPOCH = os.path.join(ROOT, 'model/modelFCNet/LfwNet/RReLU', name_model) 
        locals()[FCNetName].load_state_dict(torch.load(FCNet_PATH_EPOCH))
        fullNet.append(locals()[FCNetName])
    return fullNet

class TCNN: 
    def __init__(self, bestVanillaIndex, gmmModelPath, bestTweakedNetIndex):
        self.vanilla = loadFeatureExtractNet(bestVanillaIndex)
        self.gmmModel = joblib.load(gmmModelPath)
        self.full = loadFullNet(bestTweakedNetIndex)
    def predict(self, input):
        x = self.vanilla(input)
        x_numpy = x.cpu().data.numpy() 
        x1 = self.gmmModel.predict(x_numpy)
        ind_full = x1.squeeze()
        ind_full = torch.from_numpy(ind_full)
        y = self.full[ind_full](x)
        return y

class FcTCNN: 
    def __init__(self, bestVanillaIndex, gmmModelPath, bestTweakedNetIndex):
        self.vanilla = loadFeatureExtractNet(bestVanillaIndex)
        self.gmmModel = joblib.load(gmmModelPath)
        self.full = loadFullNet(bestTweakedNetIndex)
    def predict(self, input):
        x , y = self.vanilla(input)
        x_numpy = y.cpu().data.numpy() 
        x1 = self.gmmModel.predict(x_numpy)
        ind_full = x1.squeeze()
        ind_full = torch.from_numpy(ind_full)
        y = self.full[ind_full](x)
        return y

def loadMyNet(ind_model): 
    MyNet = MyNetBaseRReLU() 
    MyNet.cuda()
    modelName = 'MyNet_epoch%d.pkl' % (ind_model)
    modelPath = os.path.join(ROOT, 'model/modelmyNet/LfwNet/RReLU', modelName) 
    MyNet.load_state_dict(torch.load(modelPath))
    return MyNet

import matplotlib.image as mpimg
from PIL import Image

# output 1*5*40*40 
def getFeatureMapOfEachImage(image, center, sigma, width, height):
    meanGray = cv2.cvtColor(MEAN_TRAIN_SET, cv2.COLOR_RGB2GRAY)
    stdGray = cv2.cvtColor(STD_TRAIN_SET, cv2.COLOR_RGB2GRAY)
    featureMap = np.zeros([1, 5, width, height], dtype='float32')
    featureMap[0, 0:3, :, :] = image
    featureMap[0, 3:5, :, :] = guassianKernel(center, 2, 0, sigma, width, height, meanGray, stdGray)
    featureMapTensor = torch.from_numpy(featureMap)
    return featureMapTensor

class FinalNet: 
    def __init__(self, bestVanillaIndex, gmmModelPath, bestTweakedNetIndex, bestMyNetIndex):
        self.TCNN = TCNN(bestVanillaIndex, gmmModelPath, bestTweakedNetIndex)
        self.MyNet = loadMyNet(bestMyNetIndex)

    def predict(self, input):
        x = self.TCNN.predict(input)
        x_scaled = (x + 0.5) * 40
        input = input.cpu().data.numpy()
        center = x_scaled.cpu().data.numpy().squeeze()
        featureMap = getFeatureMapOfEachImage(input, center, 2, 40, 40)
        featureMap = featureMap.to(device=device, dtype=torch.float)
        y = self.MyNet(featureMap)
        z = x
        z[0, 4:10] = y[0, 4:10]
        return z
####################################################################test AFW

if 'createAFW_TestSet' in STEPS:
    print("Parsing AFW annoanno-v7.mat .....")
    from scipy.io import loadmat
    annotaions = loadmat(AFW_MAT_PATH)['anno']
    dataRowsAFW = []        
    for anno in annotaions:
        dataRow = DataRow.DataRowFromAFW(anno, AFW_DATA_PATH)
        if dataRow is not None:
            dataRowsAFW.append(dataRow)
    print("Finished parsing anno.mat with total rows:", len(dataRowsAFW))
    annotaions = None  # remove from memory    
    dataRowsAFW_Valid, R=getValidWithBBox(dataRowsAFW)
    print("Original AFW:",len(dataRowsAFW), "Valid Rows:", len(dataRowsAFW_Valid), " No faces at all", R.noFacesAtAll, " illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch)
    dataRowsAFW = None  # remove from Memory
    with open(AFW_TEST_PICKLE,'wb') as f:
        dump(dataRowsAFW_Valid, f)
        print("Data saved to AFWTestSet.pickle")

bestModleIndex = 19
VanillaTestError = '/home/haiduo/code/data/AFW/VanillaTestError.txt'
DEBUG = True 
if 'vanillaTestAFW' in STEPS:
    iamgeFilePath = ROOT + '/data/AFW/TestResults/vanilla'
    for bestModleIndex in range(19,20):
        name_model = 'VanillaEpoch%d.pkl' % (bestModleIndex) 
        MODEL_PATH_EPOCH = os.path.join(ROOT, 'model/modelVanillaNet/LefNetTrain/tanh/', name_model) 
        modelTest = VanillaNet() 
        modelTest.cuda()
        modelTest.load_state_dict(torch.load(MODEL_PATH_EPOCH))
        with open(AFW_TEST_PICKLE,'rb') as f: #141
            dataRowsAFW_Valid = load(f)
        testErrorAFW = ErrorAcumCUM() 
        for i, dataRow in enumerate(dataRowsAFW_Valid):
            dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
            image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
            image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
            image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
            image = image.permute(2,0,1) 
            image = image.unsqueeze(0) 
            prediction = modelTest(image)
            prediction = prediction.squeeze() 
            prediction = prediction.cpu().data.numpy()
            lm_0_5 = lm_0_5.cpu().data.numpy() 
            testErrorAFW.add(lm_0_5, prediction, i)
            dataRow40.prediction = (prediction + 0.5) * 40.
            if DEBUG:
                titleName = '%03d.jpg' % (i)
                dataRow40.saveImage(iamgeFilePath, title=titleName)
        #print("Test AFW:", testErrorAFW)
        strprint = "%d%s\n" % (bestModleIndex, testErrorAFW)
        file_error = open(VanillaTestError, 'a+')
        file_error.write(strprint)  
        #testErrorAFW.printMeanError()
        file_error.close()

TCNNTestError = '/home/haiduo/code/data/AFW/TCNNTestError.txt'
gmmPath = '/home/haiduo/code/model/gmmModel/LefNet/gmmRReLU'
DEBUG = True
if 'TCNNTestAFW' in STEPS:
    iamgeFilePath = ROOT + '/data/AFW/TestResults/Tweaked'
    for bestModleIndex in range(5,50):
        file_error = open(TCNNTestError, 'a+')
        #str1 = 'FcTcnn :%d\n'%(bestModleIndex)  
        #print (str1)
        #file_error.write(str1)
        for bestTweakedNetIndex in range(13, 14):
            modelTest = TCNN(bestModleIndex, gmmPath, bestTweakedNetIndex) 
            with open(AFW_TEST_PICKLE,'rb') as f:
                dataRowsAFW_Valid = load(f)
            testErrorAFW = ErrorAcumCUM()
            for i, dataRow in enumerate(dataRowsAFW_Valid):
                dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
                image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
                image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
                image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
                image = image.permute(2,0,1) 
                image = image.unsqueeze(0)
                prediction = modelTest.predict(image)
                prediction = prediction.squeeze()
                prediction = prediction.cpu().data.numpy()
                lm_0_5 = lm_0_5.cpu().data.numpy()
                testErrorAFW.add(lm_0_5, prediction, i)
                dataRow40.prediction = (prediction + 0.5) * 40.
                if DEBUG:
                    titleName = '%03d.jpg' % (i)
                    dataRow40.saveImage(iamgeFilePath, title=titleName)
            #print("TCNN Test Error AFW:", testErrorAFW)
            strprint = "%d%s\n" % (bestTweakedNetIndex, testErrorAFW)
            print (strprint)        
            file_error.write(strprint)  
            #testErrorAFW.printMeanError()
        file_error.close()

MyNetTestError = '/home/haiduo/code/data/AFW/MyNetTestErrorRReLU.txt'
bestModleIndex = 34
bestTweakedNetIndex = 4
DEBUG = True
if 'MyNetTestAFW' in STEPS:
    iamgeFilePath = ROOT + '/data/AFW/TestResults/MyNet'
    for bestTweakedNetIndex in range(4, 5):
        for bestMyNetIndex in range(14, 15):
            modelTest = FinalNet(bestModleIndex, gmmPath, bestTweakedNetIndex, bestMyNetIndex)
            with open(AFW_TEST_PICKLE,'rb') as f:
                dataRowsAFW_Valid = load(f)
            testErrorAFW = ErrorAcumCUM()
            for i, dataRow in enumerate(dataRowsAFW_Valid):
                dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
                image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
                image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
                image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
                image = image.permute(2,0,1) 
                image = image.unsqueeze(0)
                prediction = modelTest.predict(image)
                prediction = prediction.squeeze()
                prediction = prediction.cpu().data.numpy()
                lm_0_5 = lm_0_5.cpu().data.numpy()
                testErrorAFW.add(lm_0_5, prediction, i)
                dataRow40.prediction = (prediction + 0.5) * 40.
                if DEBUG:
                    titleName = '%03d.jpg' % (i)
                    dataRow40.saveImage(iamgeFilePath, title=titleName)
            print("MyNet Test Error AFW:", testErrorAFW)
            strprint = "%d %d %s\n" % (bestTweakedNetIndex, bestMyNetIndex, testErrorAFW)
            file_error = open(MyNetTestError, 'a+')
            file_error.write(strprint)  
            #testErrorAFW.printMeanError()
            file_error.close()

####################################################################
AFLW_ALL_PATH = '/home/haiduo/code/data/AFLW/datasets/AFLW_lists/test_GTB.csv'
AFLWDATA_PATH = '/home/haiduo/code/data/AFLW/images'
HD5_AFLW_PATH = '/home/haiduo/code/data/AFLW/aflw.hd5'

if 'createAFLW_TestSet' in STEPS:
    print('Start genAFLWSetHD5..')
    dataRowsGenTrain_CSV = createDataRowsFromCSV(AFLW_ALL_PATH, DataRow.DataRowFromNameBoxInterlaved1, AFLWDATA_PATH)
    print("Finished reading %d rows from training data. Parsing BBox...." % len(dataRowsGenTrain_CSV))
    dataRowsTrainValid,R = getValidWithBBox(dataRowsGenTrain_CSV)
    print("Original train:",len(dataRowsGenTrain_CSV), "Valid Rows:", len(dataRowsTrainValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch)
    dataRowsTrain_CSV=[]  # remove from memory 
    writeHD5(dataRowsTrainValid, HD5_AFLW_PATH , '/home/haiduo/code/data/AFLW/AFLWHD5Data.txt', MEAN_TRAIN_SET, STD_TRAIN_SET ,mirror=False)
    print('Finish genTrainSetHD5!')

########################################################
# Create the MTFL benchmark
if 'createMTFL_TrainSet' in STEPS:  
    MTFL_PATH = os.path.join(ROOT,'data/MTFL')
    CSV_MTFL = os.path.join(ROOT,'data/MTFL/training.txt')
    dataRowsMTFL_CSV  = createDataRowsFromCSV(CSV_MTFL , DataRow.DataRowFromMTFL, MTFL_PATH)
    print("Finished reading %d rows from train" % len(dataRowsMTFL_CSV))
    dataRowsMTFLValid,R = getValidWithBBox(dataRowsMTFL_CSV)
    print("Original train:",len(dataRowsMTFL_CSV), "Valid Rows:", len(dataRowsMTFLValid), " No faces at all", R.noFacesAtAll, " Illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch)
    with open(MTFL_TRAIN_SET,'wb') as f:
        dump(dataRowsMTFLValid,f)
    print("Finished dumping to trainSetMTFL.pickle")
    
if 'createMTFL_TestSet' in STEPS:  
    MTFL_PATH = os.path.join(ROOT,'data/MTFL')
    CSV_MTFL = os.path.join(ROOT,'data/MTFL/testing.txt')
    dataRowsMTFL_CSV  = createDataRowsFromCSV(CSV_MTFL , DataRow.DataRowFromMTFL, MTFL_PATH)
    print("Finished reading %d rows from test" % len(dataRowsMTFL_CSV))
    dataRowsMTFLValid,R = getValidWithBBox(dataRowsMTFL_CSV)
    print("Original test:",len(dataRowsMTFL_CSV), "Valid Rows:", len(dataRowsMTFLValid), " No faces at all", R.noFacesAtAll, " Illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch)
    with open(MTFL_TEST_SET,'wb') as f:
        dump(dataRowsMTFLValid,f)
    print("Finished dumping to testSetMTFL.pickle")

VanillaTestError = '/home/haiduo/code/data/MTFL/VanillaTestError.txt'
DEBUG = True
if 'VanillaTestMTFL_AFLW' in STEPS:
    iamgeFilePath = ROOT + '/data/MTFL/TestResults/Vanilla'
    for bestModleIndex in range(50,51):
        name_model = 'VanillaEpoch%d.pkl' % (bestModleIndex)
        MODEL_PATH_EPOCH = os.path.join(ROOT, 'model/modelVanillaNet/LefNetTrain/tanh/', name_model)
        modelTest = VanillaNet()  
        modelTest.cuda()
        modelTest.load_state_dict(torch.load(MODEL_PATH_EPOCH))
        with open(MTFL_TEST_SET, 'rb') as f: #2523
            dataRowsAFLW_Valid = load(f)
        testErrorAFLW = ErrorAcumCUM()
        for i, dataRow in enumerate(dataRowsAFLW_Valid):
            dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
            image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
            image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
            image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
            image = image.permute(2,0,1) 
            image = image.unsqueeze(0)
            prediction = modelTest(image)
            prediction = prediction.squeeze()
            prediction = prediction.cpu().data.numpy()
            lm_0_5 = lm_0_5.cpu().data.numpy() 
            testErrorAFLW.add(lm_0_5, prediction, i)
            dataRow40.prediction = (prediction + 0.5) * 40.
            if DEBUG:
                titleName = '%03d.jpg' % (i)
                dataRow40.saveImage(iamgeFilePath, title=titleName)
        #print("Vanilla Test Error MTFL_AFLW:", testErrorAFLW)
        strprint = "%d%s\n" % (bestModleIndex, testErrorAFLW)
        file_error = open(VanillaTestError, 'a+')
        file_error.write(strprint)
        #testErrorAFLW.printMeanError()
        file_error.close()  

TCNNTestError = '/home/haiduo/code/data/MTFL/TCNNTestError.txt'
gmmPath = '/home/haiduo/code/model/gmmModel/LefNet/gmmRReLU'
DEBUG = True
if 'TCNNTestMTFL_AFLW' in STEPS:
    iamgeFilePath = ROOT + '/data/MTFL/TestResults/Tweaked'
    for bestModleIndex in range(36, 37):
        file_error = open(TCNNTestError, 'a+')
        str1 = 'vanilla:%d\n'%(bestModleIndex)
        print (str1)
        file_error.write(str1)
        for bestTweakedNetIndex in range(2,3):
            modelTest = TCNN(bestModleIndex, gmmPath, bestTweakedNetIndex) 
            with open(MTFL_TEST_SET, 'rb') as f:
                dataRowsAFLW_Valid = load(f)
            testErrorAFLW = ErrorAcumCUM()
            for i, dataRow in enumerate(dataRowsAFLW_Valid):
                dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
                image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
                image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
                image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
                image = image.permute(2,0,1) 
                image = image.unsqueeze(0)
                prediction = modelTest.predict(image)
                prediction = prediction.squeeze()
                prediction = prediction.cpu().data.numpy()
                lm_0_5 = lm_0_5.cpu().data.numpy()
                testErrorAFLW.add(lm_0_5, prediction, i)
                dataRow40.prediction = (prediction + 0.5) * 40.
                if DEBUG:
                    titleName = '%03d.jpg' % (i)
                    dataRow40.saveImage(iamgeFilePath, title=titleName)
            #print("TCNN Test Error MTFL_AFLW:", testErrorAFLW)
            strprint = "%d%s\n" % (bestTweakedNetIndex, testErrorAFLW)
            print (strprint)        
            file_error.write(strprint)
            #testErrorAFLW.printMeanError()
        file_error.close()

MyNetTestError = '/home/haiduo/code/data/MTFL/MyNetTestErrorRReLU.txt'
bestModleIndex = 37 
bestTweakedNetIndex = 2
DEBUG = True
if 'MyNetTestMTFL_AFLW' in STEPS:
    iamgeFilePath = ROOT + '/data/MTFL/TestResults/MyNet'
    for bestTweakedNetIndex in range(2, 3):
        for bestMyNetIndex in range(20, 21):
            modelTest = FinalNet(bestModleIndex, gmmPath, bestTweakedNetIndex, bestMyNetIndex)
            with open(MTFL_TEST_SET,'rb') as f:
                dataRowsAFLW_Valid = load(f)
            testErrorAFLW = ErrorAcumCUM()
            for i, dataRow in enumerate(dataRowsAFLW_Valid):
                dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
                image, lm_0_5 = preprocess(dataRow40.image, dataRow40.landmarks())
                image, lm_0_5 = torch.from_numpy(image), torch.from_numpy(lm_0_5)
                image, lm_0_5 = image.to(device=device, dtype=torch.float), lm_0_5.to(device=device, dtype=torch.float)
                image = image.permute(2,0,1) 
                image = image.unsqueeze(0)
                prediction = modelTest.predict(image)
                prediction = prediction.squeeze()
                prediction = prediction.cpu().data.numpy()
                lm_0_5 = lm_0_5.cpu().data.numpy()
                testErrorAFLW.add(lm_0_5, prediction, i)
                dataRow40.prediction = (prediction + 0.5) * 40.
                if DEBUG:
                    titleName = '%03d.jpg' % (i)
                    dataRow40.saveImage(iamgeFilePath, title=titleName)
            #print("MyNet Test Error MTFL_AFLW:", testErrorAFLW)
            strprint = "%d %d %d %s\n" % (bestModleIndex, bestTweakedNetIndex, bestMyNetIndex, testErrorAFLW)
            print (strprint)
            file_error = open(MyNetTestError, 'a+')
            file_error.write(strprint)  
            #testErrorAFLW.printMeanError()
            file_error.close()
