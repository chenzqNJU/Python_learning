import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import tushare as ts
import os
import gc
import myfunc
os.chdir("e:\\stock\\siyinzi")
myfunc.search('sas',1)

#导入前复权收盘价
close=pd.read_csv('closeADJ.csv')
close=close.iloc[:,1:]
close.ticker=close.ticker.astype(str)
close.ticker=close.ticker.str.zfill(6)
close1=pd.pivot_table(close,columns='ticker',values='closePrice',index='tradeDate')
mad = lambda x: x[1]/x[0]-1
close2=close1.rolling(window=2).apply(mad)
close=close2.stack()
close.name='rate'
close=close.reset_index()

#导入市值 PB
shizhi_PE=pd.read_csv('shizhi_PE.csv')
shizhi_PE=shizhi_PE.iloc[:,1:]
shizhi_PE.ticker=shizhi_PE.ticker.astype(str)
shizhi_PE.ticker=shizhi_PE.ticker.str.zfill(6)
shizhi_PE['BM']=1/shizhi_PE.PB

#shizhi_PE[shizhi_PE.ticker=='002839']


f=pd.merge(close,shizhi_PE,on=['ticker','tradeDate'])
f=f.drop('PB',axis=1)
#导入沪深300
hs300=pd.read_csv('hs300.csv')
hs300=hs300.iloc[:,2:]
mad = lambda x: x[1]/x[0]-1
hs300=hs300.rolling(window=2).apply(mad)

f=pd.merge(f,hs300,on=['tradeDate'])
f.set_index('tradeDate',inplace=True)

#######################################################5因子 导入盈利因子 投资因子
factor5=pd.read_csv('factor5.csv')
factor5=factor5.iloc[:,1:]
factor5.ticker=factor5.ticker.astype(str)
factor5.ticker=factor5.ticker.str.zfill(6)

f=f.reset_index()
f=pd.merge(f,factor5,on=['tradeDate','ticker'])
f.set_index('tradeDate',inplace=True)

###分组后组间收益差
#市值中位数
median_size = {}
for date in set(f.index):
    median_size[date] = np.median(f['negMarketValue'].loc[date])
median_size=pd.Series(median_size)
median_size=median_size.to_frame(name='p50')

#BM的30% 70%
BM=pd.DataFrame(index=median_size.index,columns=['p30','p70'])
f1=f[f.BM>0]
for date in BM.index:
    BM.p30[date] = np.percentile(list(f1.BM.loc[date]),30)
    BM.p70[date] = np.percentile(list(f1.BM.loc[date]),70)
############################################# OperatingProfitGrowRate TotalAssetGrowRate的30% 70%
OperatingProfitGrowRate=pd.DataFrame(index=median_size.index,columns=['p30','p70'])
f1=f[~f.OperatingProfitGrowRate.isnull()]
for date in OperatingProfitGrowRate.index:
    OperatingProfitGrowRate.p30[date] = np.percentile(list(f1.OperatingProfitGrowRate.loc[date]),30)
    OperatingProfitGrowRate.p70[date] = np.percentile(list(f1.OperatingProfitGrowRate.loc[date]),70)
TotalAssetGrowRate=pd.DataFrame(index=median_size.index,columns=['p_30','p_70'])
f1=f[~f.TotalAssetGrowRate.isnull()]
for date in TotalAssetGrowRate.index:
    TotalAssetGrowRate.p_30[date] = np.percentile(list(f1.TotalAssetGrowRate.loc[date]),30)
    TotalAssetGrowRate.p_70[date] = np.percentile(list(f1.TotalAssetGrowRate.loc[date]),70)
f2=f.join([OperatingProfitGrowRate,TotalAssetGrowRate])
#将 市值 BM分类
f2.loc[f2.OperatingProfitGrowRate<f2.p30,'rmw']=0
f2.loc[f2.OperatingProfitGrowRate>=f2.p70,'rmw']=2
f2.rmw.fillna(1,inplace=True)
f2.loc[f2.TotalAssetGrowRate<f2.p30,'cma']=0
f2.loc[f2.TotalAssetGrowRate>=f2.p70,'cma']=2
f2.cma.fillna(1,inplace=True)

f2.rmw=f2.rmw.astype(int)
f2.cma=f2.cma.astype(int)
###分类计算
t=f2[['rate','rmw','cma']]
t.index.name='tradeDate'
t1=t.groupby(['tradeDate','rmw'])['rate'].agg('mean')
t1=t1.unstack()
t1['RMW']=t1[2]-t1[0]

t2=t.groupby(['tradeDate','cma'])['rate'].agg('mean')
t2=t2.unstack()
t2['CMA']=t2[0]-t2[2]
t=pd.concat([t1.loc[:,['RMW']],t2.loc[:,['CMA']]],axis=1)

final=pd.read_csv('sanyinzi.csv')
final.ticker=final.ticker.astype(str)
final.ticker=final.ticker.str.zfill(6)

final.rename(columns={'Unnamed: 0':'tradeDate'}, inplace = True)
final.set_index('tradeDate',inplace=True)
final=final.join(t)

final.to_csv('wuyinzi.csv')
#######################################################
f2=f.join([median_size,BM])
#将 市值 BM分类
f2.loc[f2.negMarketValue<f2.p50,'mv']=0
f2.loc[f2.negMarketValue>=f2.p50,'mv']=1

f2.loc[(f2.BM<f2.p30)&(f2.BM>0),'bm']=0
f2.loc[f2.BM>f2.p70,'bm']=2
f2.bm.fillna(1,inplace=True)

f2.mv=f2.mv.astype(int)
f2.bm=f2.bm.astype(int)



###分类计算
t=f2[['rate','mv','bm']]
t.index.name='tradeDate'
t1=t.groupby(['tradeDate','mv'])['rate'].agg('mean')
t1=t1.unstack()
t1['SMB']=t1[1]-t1[0]
t2=t.groupby(['tradeDate','bm'])['rate'].agg('mean')
t2=t2.unstack()
t2['HML']=t2[0]-t2[2]
t=pd.concat([t1.loc[:,['SMB']],t2.loc[:,['HML']]],axis=1)
f3=f2.join(t)

lilv=pd.DataFrame()
for x in range(2008,2019):
    df = ts.shibor_data(x)
    lilv=lilv.append(df)
lilv=lilv.iloc[:,[0,-1]]

lilv.index=[x.strftime("%Y-%m-%d") for x in lilv.date]
lilv['loan']=lilv['1Y']/12/100
lilv['loan']=lilv['loan']/100
lilv=lilv.loc[:,['loan']]
f4=f3.join(lilv)
f4['MKT']=f4.closeIndex-f4.loan
################
f5=f4[['ticker','rate','MKT','SMB','HML']]
f5.to_csv('sanyinzi.csv')


len(set(f5.index))
a=list(set(f5.index))
a.sort()
a[60]

len(f5[f5.ticker=='000001'])
a=f5[f5.ticker.str.startswith('2')]
len(set(a.ticker))
#######################################
#######################################
#######################################
#######################################ROE
ROE=pd.read_csv('ROE.csv')
ROE=ROE.iloc[:,1:]
ROE.ticker=ROE.ticker.astype(str)
ROE.ticker=ROE.ticker.str.zfill(6)
ROE['year']=ROE.endDate.str.slice(0,4)
ROE.year=ROE.year.astype(int)
ROE=ROE[(ROE.year<2018) & (ROE.year>2012)]
ROE1=ROE.drop_duplicates(subset=['ticker','endDate'],keep='last')
ROE1=ROE1.groupby(['ticker','year'])['ROE'].agg('std')
ROE1=ROE1.to_frame()
ROE1.reset_index(inplace=True)
t=ROE1.groupby(['year'])['ROE'].agg('median')
t=t.to_frame(name='ROE_m')
t.reset_index(inplace=True)
ROE2=pd.merge(ROE1,t,how='left')
ROE2['DROE']=ROE2.ROE/ROE2.ROE_m
ROE2=ROE2.drop(['ROE','ROE_m'],axis=1)
#######################################资产周转率
zc=pd.read_csv('zichanzhouzhuanlv.csv')
zc=zc[zc.endDate.str.endswith('12-31')]
zc=zc.iloc[:,1:]
zc.ticker=zc.ticker.astype(str)
zc.ticker=zc.ticker.str.zfill(6)
zc=zc.drop_duplicates(subset=['ticker','endDate'],keep='last')
zc['year']=zc.endDate.str.slice(0,4).astype(int)
zc=zc[(zc.year<2018) & (zc.year>2012)]
zc=zc.drop('endDate',axis=1)
#######################################流动比率
ld=pd.read_csv('liudongbilv.csv')
ld=ld.iloc[:,1:]
ld.ticker=ld.ticker.astype(str)
ld.ticker=ld.ticker.str.zfill(6)
ld=ld.drop_duplicates(subset=['ticker','endDate'],keep='last')
ld['year']=ld.endDate.str.slice(0,4).astype(int)
ld=ld[(ld.year<2018) & (ld.year>2012)]
ld=ld.groupby(['ticker','year'])['currenTRatio'].agg('mean')
ld=ld.to_frame()
ld.reset_index(inplace=True)
#######################################PB
shizhi_PE=pd.read_csv('shizhi_PE.csv')
shizhi_PE=shizhi_PE.iloc[:,1:]
shizhi_PE.ticker=shizhi_PE.ticker.astype(str)
shizhi_PE.ticker=shizhi_PE.ticker.str.zfill(6)
shizhi_PE=shizhi_PE.sort_values(['ticker','tradeDate'])

shizhi_PE['year']=shizhi_PE.tradeDate.str.slice(0,4).astype(int)
shizhi_PE=shizhi_PE[(shizhi_PE.year<2018) & (shizhi_PE.year>2012)]
shizhi_PE=shizhi_PE.groupby(['ticker','year'])['PB'].agg('mean')
shizhi_PE=shizhi_PE.to_frame()
shizhi_PE.reset_index(inplace=True)
#######################################资产负债率
le=pd.read_csv('zichanfuzhailv.csv')
le=le.iloc[:,1:]
le.ticker=le.ticker.astype(str)
le.ticker=le.ticker.str.zfill(6)

le['year']=le.endDate.str.slice(0,4).astype(int)
le=le[(le.year<2018) & (le.year>2012)]
le=le.drop_duplicates(subset=['ticker','endDate'],keep='last')


le=le.groupby(['ticker','year'])['asseTLiabRatio'].agg('mean')
le=le.to_frame()
le.reset_index(inplace=True)
#######################################合并
t1=pd.merge(ROE2,zc,on=['ticker','year'],how='outer')
t1=pd.merge(t1,ld,on=['ticker','year'],how='outer')
t1=pd.merge(t1,shizhi_PE,on=['ticker','year'],how='outer')
t1=pd.merge(t1,le,on=['ticker','year'],how='outer')
t1.to_csv('zhibiao_all.csv')


help(pd.merge)

z=ROE1[ROE1.year==2013]
z.ROE.mean()
a=z.ROE.values
np.nanmax(a)

z=ROE1[ROE1.ROE>10000]
z=ROE[ROE.ticker=='000035']

myfunc.search('year')

help(pd.DataFrame.drop_duplicates)

data.drop_duplicates()

############################################### 股市崩盘
#导入前复权 周收盘价
close=pd.read_csv('zhou_close.csv')
close=close.iloc[:,1:]
close.ticker=close.ticker.astype(str)
close.ticker=close.ticker.str.zfill(6)
a=close.closePrice.values
b=np.round((a[1:]-a[:-1])/a[:-1],4)
close['return']=[0]+b.tolist()

#导入沪深300 日度
hs=pd.read_csv('hushen300.csv')
hs=hs.iloc[:,2:]
hs.set_index('tradeDate',inplace=True)
t=list(set(close.endDate))
t.sort()
hs=hs.loc[t,:]

a=hs.closeIndex.values
b=np.round((a[1:]-a[:-1])/a[:-1],4)
hs['ret']=[0]+b.tolist()

hs.to_csv('hushen300_.csv')








