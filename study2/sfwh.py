'''
实丰文化研究
'''
from queue import Queue
import threading
import os
import datetime
import pandas as pd
import tushare as ts
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import types
import myfunc
import pprint
from imp import reload
reload(myfunc)

s=pd.Series(np.random.randn(50))
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
df.sample(frac=1)


train_val_df.sample(frac=1)

help(pd.DataFrame.sample)

dates = myfunc.get_date_list()[-10:]
code = '002862'

df = pd.DataFrame()
for date in dates:
    try:
        qfq_day = ts.get_tick_data(code, date, pause=0.1, src='tt')
        qfq_day['code'] = code
        qfq_day['date'] = date
        df =df.append(qfq_day)
        print(date)
    except:
        # task_qeue.put(data)  # 如果数据获取失败，将该数据重新存入到队列，便于后期继续执行
        print(code + '  ' + date + '  None')
d=df.copy()
df=d
############################ 整理成1,5,10,30,60 分钟形式，9:30-9:34 的形式，11:30，15:00归入前一类

def change_min(df,min=1):
    df['time'] = [x.time() for x in pd.to_datetime(df.time, format='%H:%M:%S')] # 字符转日期，转日间
    df.time = df.time.apply(lambda x: x.replace(second=0))          #秒为0
    df1 = df.groupby(['date','time'])['volume', 'amount'].sum()
    df2 = df.groupby(['date','time'])['price'].last()
    df = df1.join(df2)
    if min==1:return df

    df.reset_index(inplace=True)
    #### 将9.25集合竞价的归到9.30
    df.loc[df.time==datetime.time(9, 25),'time']=datetime.time(9, 30)
    df['datetime'] = pd.to_datetime(df.date + ' ' + df.time.astype(str))

    #### mod分类
    df['min'] = np.mod(df.datetime.dt.minute, min) * datetime.timedelta(minutes=1)
    if min==60:
        df.loc[df.datetime.dt.hour < 12,'min'] = np.mod(df.datetime.dt.minute+30, min) * datetime.timedelta(minutes=1)

    ###### 将3:00的归到2:55-2:59 11:30的归到
    tmp1 = df.datetime.dt.hour == 15
    tmp2 = (df.datetime.dt.hour == 11) & (df.datetime.dt.minute == 30)
    df.loc[(tmp1 | tmp2), 'min'] = min * datetime.timedelta(minutes=1)

    df.datetime = df.datetime - df['min']
    del df['min']

    # 高开低收
    df1 = df.groupby('datetime')['volume', 'amount'].sum()
    df2 = df.groupby('datetime')['price'].last();df2.name='close'
    df3 = df.groupby('datetime')['price'].first();df3.name = 'open'
    df4 = df.groupby('datetime')['price'].max();df4.name = 'high'
    df5 = df.groupby('datetime')['price'].min();df5.name = 'low'
    df = df1.join(df2).join(df3).join(df4).join(df5)
    return df

def kdj(df):

    df['RSV'] = np.round((df.close - df.low) / (df.high - df.low) * 100,2)
    df.RSV.replace(np.inf, np.nan, inplace=True)
    # 涨跌停
    temp = Price[:1].copy();
    temp[:] = np.nan
    temp1 = round((Price + 0.000001) * 1.1, 2)
    temp1 = temp.append(temp1[:-1])
    temp1.index = Price.index
    zhangting = temp1 == Price
    temp2 = round((Price + 0.000001) * 0.9, 2)
    temp2 = temp.append(temp2[:-1])
    temp2.index = Price.index
    dieting = (temp2 == Price)

    RSV[dieting] = 0
    RSV[zhangting] = 100
    ####计算k值
    for n, i in enumerate(RSV.index):
        # n=1;i=Price.index[n]
        if n == 0:
            K = np.ones((1, RSV.shape[1])) * 50;
            tempK = K[0].copy()
            D = K.copy()
            tempD = tempK.copy()
            RSV_ = RSV.values
        else:
            x = np.where(isOpen.loc[i] == 1)
            tempK[x] = RSV_[n, x] * 1 / 3 + tempK[x] * 2 / 3
            # temp[np.isnan(temp)^np.isnan(Price_[n,])]=50
            K = np.row_stack((K, tempK))
            tempD[x] = tempK[x] * 1 / 3 + tempD[x] * 2 / 3
            D = np.row_stack((D, tempD))

    K = pd.DataFrame(K, index=Price.index, columns=Price.columns).round(2)
    D = pd.DataFrame(D, index=Price.index, columns=Price.columns).round(2)
    K_ = pd.DataFrame(K.unstack(), columns=['K'])
    D_ = pd.DataFrame(D.unstack(), columns=['D'])
    kdj = pd.concat([K_, D_], axis=1)
    kdj['J'] = kdj.K * 2 - kdj.D

df = change_min(d,5)


pprint.pprint(a)

df.time[1].values

df.datetime.dt.minute
min=60

qfq_day.time[1]