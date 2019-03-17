# coding = utf-8
import numpy as np
import math

t=np.arange(0,1,0.1)
t1=t.T
np.linspace(0,10)
x=np.arange(0,np.pi/2,0.1)
y=np.sin(x)

import numpy as np
import matplotlib.pyplot as plt

a=np.pi
x=np.arange(-np.pi,np.pi,0.01)
y=np.sin(x)
plt.plot(x,y,label="dd",color='c',linewidth=2)
plt.title('打得到',fontproperties=font_set)
plt.axis([-3,3,-1,1])
plt.grid(True)
plt.savefig("test.png",dpi=120)
plt.show()
plt.figure(figsize=(8,4))
plt.plot(x,y,label="$sin(x)^2$",color='c',linewidth=2)
plt.plot(x,y-1,label="$sin(x)^2$",color='c',linewidth=2)

plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.5,1.5)
plt.show()
run matplotlib_simple_plot.py


plt.figure(1) # 创建图表1
plt.figure(2) # 创建图表2
ax1 = plt.subplot(211) # 在图表2中创建子图1
ax2 = plt.subplot(212) # 在图表2中创建子图2

x = np.linspace(0, 3, 100)
for i in xrange(5):
	plt.figure(1) # 选择图表1
	plt.plot(x, np.exp(i*x/3))
	plt.sca(ax1) # 选择图表2的子图1
	plt.plot(x, np.sin(i*x))
	plt.sca(ax2) # 选择图表2的子图2
	plt.plot(x, np.cos(i*x))
plt.show()





from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2, 3), (0, 0))
sex_group.plot(kind='bar')
plt.title(u'性别分布', fontproperties=font_set)
plt.ylabel(u'人数', fontproperties=font_set)






