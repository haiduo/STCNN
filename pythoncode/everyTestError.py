import numpy as np
import matplotlib.pyplot as plt

vanillaTestErrorPath = '/home/haiduo/code/data/MTFL/VanillaTestError.txt'
TweakedTestErrorPath = '/home/haiduo/code/data/MTFL/MyNetTestError.txt'
myNetTestErrorPath = '/home/haiduo/code/data/MTFL/MyNetTestError.txt'
#1mean error:9.649678% [0.08242951 0.07694426 0.13302809 0.09276726 0.09731478] 141 items,50 failures(>0.1) accuracy:64.539007%
epoch = []
vanillaTestError = []
TweakedTestError = []
myNetTestError = []

TweakedTestAccuracy = []
TweakedTestEveryAccuracy = []
# with open(vanillaTestErrorPath) as f:
#     for line in f:
#         line = line.split(':')
#         epoch.append(int((line[0].split('mean'))[0]))
#         vanillaTestError.append(float((line[1].split('%'))[0]))
# with open(TweakedTestErrorPath) as f:
#     for line in f:
#         line = line.split(':')
#         TweakedTestError.append(float((line[1].split('%'))[0]))
# with open(myNetTestErrorPath) as f:
#     for line in f:
#         line = line.split(':')
#         myNetTestError.append(float((line[1].split('%'))[0]))

with open(TweakedTestErrorPath) as f:
    for line in f:
        line = line.split(':')
        epoch.append(int((line[0].split('mean'))[0]))
        TweakedTestError.append(float((line[1].split('%'))[0]))
        #TweakedTestAccuracy.append(float((line[2].split('%'))[0]))
        TweakedTestEveryAccuracy.append(list(map(float,(line[1].split('[')[1].split(']')[0].split( ))[:])))

epoch = np.array(epoch)
# vanillaTestError = np.array(vanillaTestError)
TweakedTestError = np.array(TweakedTestError)
# myNetTestError = np.array(myNetTestError)

#TweakedTestAccuracy = np.array(TweakedTestAccuracy) #less than 0.1
TweakedTestEveryAccuracy = np.array(TweakedTestEveryAccuracy)*100

plt.figure(num = 'MyNetTestAFLWerror', figsize = (8,6), dpi = 80)
ax = plt.subplot(1,1,1)
plt.title('MyNetTestAFLWerror')

# plt.plot(epoch, vanillaTestError[:], 'r', label='vanilla', linewidth=1)
plt.plot(epoch, TweakedTestError[:], 'r', label='Tweaked', linewidth=1)
# plt.plot(epoch, myNetTestError[:], 'b', label='myNet', linewidth=1)
# plt.plot(epoch, TweakedTestAccuracy[:,0], 'b', label='l.eye', linewidth=1)

plt.plot(epoch, TweakedTestEveryAccuracy[:,0], 'b', label='l.eye', linewidth=1)
plt.plot(epoch, TweakedTestEveryAccuracy[:,1], 'b', label='r.eye', linewidth=1)
plt.plot(epoch, TweakedTestEveryAccuracy[:,2], 'g', label='nose', linewidth=1)
plt.plot(epoch, TweakedTestEveryAccuracy[:,3], 'm', label='l.mouth', linewidth=1)
plt.plot(epoch, TweakedTestEveryAccuracy[:,4], 'm', label='r.mouth', linewidth=1)


plt.xlim(epoch.min()*0.1, epoch.max()*1.01)
plt.ylim(np.array([TweakedTestEveryAccuracy.min()]).min()*0.95, np.array([TweakedTestEveryAccuracy.max()]).max()*1.1)

plt.xlabel(u'ModelEpoch')
plt.ylabel(u'AFLW Error(%)')
plt.legend()
figPath = '/home/haiduo/code/data/MTFL/MyNetTestMTFLerror.png'
plt.savefig(figPath, dpi=80)
plt.show()

















