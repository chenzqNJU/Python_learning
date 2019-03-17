import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
#定义名字，为了，简单命名，第一列是序号，最后一列是输出
column_names=['sample code number','1','2','3','4','5','6','7','8','9','class']
#利用pandas从网上下载数据
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
#删除丢失的不完整的数据
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
data.shape
x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
y_train.value_counts()
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
lr=LogisticRegression()
sgdc=SGDClassifier()
lr.fit(x_train,y_train)
lr_y_predict=lr.predict(x_test)
sgdc.fit(x_train,y_train)
sgdc_y_predict=sgdc.predict(x_test)
print ("Accuracy of LR Classifier:",lr.score(x_test,y_test))
print (classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))
print "finish"