import numpy as np
import numpy.linalg as nplg
t1=np.zeros((2,2))
t2=np.ones_like(t1)
t3=[t1,t2]

t1[1,:]
t1+t2
t1>0
t3=t2*2
t2/t3
t2.sum(axis=1)
t3.min(axis=0)
np.sin(t2)
np.dot(t2,t3)
#合并
np.vstack((t2,t3))
t4=np.concatenate((t1,t2),axis=1)
t5=np.split(t4,2,axis=1)
t5[1]
#矩阵
a = np.ones((2,2))*2
b = a
a[1,1]=4
b is a
b=a.copy()

a=np.array([[1,0],[2,3]])
a.transpose()
#多维转一维
a.flatten().reshape(2,2)

a.ravel() # 平坦化数组