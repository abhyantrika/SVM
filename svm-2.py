from sklearn import svm,datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns

data2 = sio.loadmat('data2.mat')

x = data2['X']
y = data2['y']

h = 0.01


model = svm.SVC(kernel ='rbf',C=1,gamma = 30)
model.fit(x,y) 									# HIGH GAMMA NEEDED TO FIT!!!
print 'Score is','='*10+'>',model.score(x,y)

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

