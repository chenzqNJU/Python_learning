import pandas as pd
import numpy as np
import time
data=pd.read_excel('C:\\Users\\chenzq\\Documents\\WeChat Files\\czq13851976255\\Files\\以前\\5秒级别slope计算\\中铁二局output.xlsx')

data1=data
stockname=data.stockname[1]
stockcode=data.stockcode[1];
date=data.tradedate[1]
print(date)
t1=date.to_datetime
t1=pd.to_datetime(date)
t=time.mktime(date)
dt = "2016-05-05 20:28:54"
time_tuple = time.localtime(date)
#转换成时间数组
timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
#转换成时间戳
timestamp = time.mktime(timeArray)

date=pd.Series.drop_duplicates(data.tradedate,keep='first',inplace=False)
date=pd.Series.drop_duplicates(data.tradedate, keep='first',inplace=False)
z=date.tolist()
date.values

date.index=range(22)

z1=date[4]

# 删除不用的变量
dropVariables=['p'+str(x+1) for x in range(5)]+['stockcode','stockname','pipei','tradetimep','deltaP']
data.pop(dropVariables)
data.drop(dropVariables,1,inplace=True)

#data.columns[[0,1]]=['sellV','buyV']

data.rename(columns={'v1' : 'sellV','v2':'buyV'},inplace=True)
Samount=data.iloc[:,12:17].values
Samount=np.fliplr(Samount)
CumSamount=np.log(np.cumsum(Samount,1))

Bamount=data.iloc[:,17:22].values
CumBamount=np.log(np.cumsum(Bamount,1))

#计算公式中的分母项，挂单高档位较低档位的变化幅度。
Sprice=data.iloc[:,6:1:-1].values
any(np.where(Sprice==0))
PctSprice=(Sprice[:,1:]-Sprice[:,:-1])/Sprice[:,:-1]


Bprice=data.iloc[:,7:12].values
any(np.where(Bprice==0)[0])
index=np.where(Bprice[:,:-1]==0)
Bprice[index]=1
PctBprice=(Bprice[:,1:]-Bprice[:,:-1])/Bprice[:,:-1]
PctBprice[index]=np.nan
#计算分子
# python 累加是横向累加，matlab累加是纵向的
PctCumSamount=(CumSamount[:,1:]-CumSamount[:,:-1])/CumSamount[:,:-1]
PctCumBamount=(CumBamount[:,1:]-CumBamount[:,:-1])/CumBamount[:,:-1]
#计算各个时点的slope
slope=np.mean(PctCumSamount/PctSprice + abs(PctCumBamount/PctBprice),1).T/2
data['slope']=slope
final=data[['tradedate','slope']]
final1=final.groupby('tradedate').mean()


final1[final1.slope>100]
final1['date']=final1.index
final1['date1']=[x.strftime("%Y-%m-%d") for x in date]

final1[[2]]
final[final.isnull().values==True]

aa=[x.strftime("%Y-%m-%d") for x in date]
aa=date[1]
z.dtype=float
z[1,1]=6.6

 import datetime
 date_str = "2016-11-30 13:53:59"
 datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
datetime.datetime(2016, 11, 30, 13, 53, 59)

import datetime
dt_obj = datetime.datetime(2016, 11, 30, 13, 53, 59)

np.mean(z,1)
a=np.array([0,0,1,np.nan])
np.nanmean(a)
any(np.where(Bprice==0))
zz1=np.argwhere(Bprice==0)
zz=np.where(Bprice==0)
any(zz1[0])
np.where(Sprice==0) is None

Bprice[zip(zz1[:,0],zz1[:,1])]

Bprice[([1,2],[2,2])]
zz2=tuple(zz1.T)
len(zz)
Bprice[zz2]
zz[1]


z is None
z=np.array([[1,2,3],[2,3,4]])
np.cumsum(z,1)
np.fliplr(z,inplace=True)
A = np.random.randn(2,3,5)
np.all(np.fliplr(A) == A[:,::-1,...])
z[:,::-1]

(z[:,1:]-z[:,:-1])/z[:,:-1]

z[:,::-1]
z[::-1,:]

a = np.array(([3, 2, 1], [2, 5, 7], [4, 7, 8]))

a1=np.argwhere(a == 7)
a2=np.where(a == 7)
a1[0,1]
data.columns.str.strip('s')

z1=(np.arange(5)+1)

z2=z1.tolist()
[str(x) for x in z2]
clist = ['s','t','r']
''.join( clist )

str([1,2,3])


'p'+z1
z3=map(str,z2)

dropVariables=[cellstr(strcat('p',num2str([1:5]')))','stockcode','stockname'\
    'pipei','tradetimep','deltaP']


data(:,dropVariables)=[];

 A = np.random.randn(2,3,5)
>>> np.all(np.fliplr(A) == A[:,::-1,...])

 a=np.array([[1,2,3,4],[2,3,4,5]])
 a[:,::-1]