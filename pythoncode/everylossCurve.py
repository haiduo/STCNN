import numpy as np
import matplotlib.pyplot as plt

VanillaTrainLossPath = '/home/haiduo/code/lossAndAccuracy/LefNet/VanillaTrainLoss.txt'
TweakedTrainLossPath = '/home/haiduo/code/lossAndAccuracy/LefNet/TweakedTrainLoss.txt'
myNetTrainLossPath = '/home/haiduo/code/lossAndAccuracy/LefNet/VanillaTrainLoss.txt'
#epoch:1 trainloss:0.070 trainAccuracy:0.02475 validateAccuracy:0.07295 trainEvery:0.02121 0.02060 0.02834 0.02598 0.02762 validEvery:0.06278 0.06037 0.08365 0.07658 0.08135
epoch = []
myNetTrainAccuracy = []
myNetValidateAccuracy = []
with open(myNetTrainLossPath) as f:
    for line in f:
        line = line.split(':')
        epoch.append(int((line[1].split( ))[0]))
        myNetTrainAccuracy.append(list(map(float,(line[5].split( ))[:-1])))
        myNetValidateAccuracy.append(list(map(float,(line[6].split( ))[:])))

epoch = np.array(epoch)

myNetTrainAccuracy = np.array(myNetTrainAccuracy)*100
myNetValidateAccuracy = np.array(myNetValidateAccuracy)*100

plt.figure(num = 'Vanilla', figsize = (8,6), dpi = 80)
ax = plt.subplot(1,1,1)
plt.title('Vanilla')

plt.plot(epoch, myNetTrainAccuracy[:,0], 'b', label='l.eyeTrain', linewidth=1)
plt.plot(epoch, myNetTrainAccuracy[:,1], 'b', label='r.eyeTrain', linewidth=1)
plt.plot(epoch, myNetTrainAccuracy[:,2], 'g', label='noseTrain', linewidth=1)
plt.plot(epoch, myNetTrainAccuracy[:,3], 'm', label='l.mouthTrain', linewidth=1)
plt.plot(epoch, myNetTrainAccuracy[:,4], 'm', label='r.mouthTrain', linewidth=1)

plt.plot(epoch, myNetValidateAccuracy[:,0], 'b--', label='l.eyeValidate', linewidth=1)
plt.plot(epoch, myNetValidateAccuracy[:,1], 'b--', label='r.eyeValidate', linewidth=1)
plt.plot(epoch, myNetValidateAccuracy[:,2], 'g--', label='noseValidate', linewidth=1)
plt.plot(epoch, myNetValidateAccuracy[:,3], 'm--', label='l.mouthValidate', linewidth=1)
plt.plot(epoch, myNetValidateAccuracy[:,4], 'm--', label='r.mouthValidate', linewidth=1)

plt.xlim(epoch.min()*0.1, epoch.max()*1.01)
plt.ylim(np.array([myNetTrainAccuracy.min(), myNetValidateAccuracy.min()]).min()*0.9, np.array([myNetTrainAccuracy.max(), myNetValidateAccuracy.max()]).max()*1.01)

plt.xlabel(u'epoch')
plt.ylabel(u'Accuacy(%)')
plt.legend()
figPath = '/home/haiduo/code/lossAndAccuracy/LefNet/VanillaTrainEveryAccuacy.png'
plt.savefig(figPath, dpi=80)
plt.show()

















