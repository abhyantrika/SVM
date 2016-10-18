from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


data = sio.loadmat('data1.mat')
x = data['X']
y = data['y']
h = 0.01

model = svm.SVC(kernel = 'linear',C =0.2, gamma = 10)
model.fit(x,y)
print model.score(x,y)

""" These are plotting stuff. Need to learn these!"""
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

plt.subplot(1,1,1)
a = np.c_[xx.ravel(),yy.ravel()]
z = model.predict(a)
z= z.reshape(xx.shape)

plt.contourf(xx,yy,z,cmap=plt.cm.Paired,alpha=1)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())

plt.show()

