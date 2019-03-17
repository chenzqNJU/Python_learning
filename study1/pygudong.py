import tushare as ts
import numpy as np
import pandas as pd

data=pd.read_csv("E:\\bank\\roa1.csv")
len(data)

a=ts.get_stock_basics()

#########表内表外利息统计
data=pd.read_excel("E:\\bank\\表内表外利息统计.xlsx")
data1=data.iloc[:,1:].values
data2=data1.flatten(1)
data3=pd.DataFrame()
data3['danwei']=np.tile(data.单位,3)

df = pd.DataFrame({'单位':np.tile(data.单位,4),
'年份':[2013]*62*2+[2014]*62*2,
 '项目':np.tile(np.repeat(['表内','表外'],62)  ,2),
'金额':data2
                   })
df1=pd.pivot_table(df,index=["项目"],values=["金额"],
               columns=["单位","年份"],fill_value=0)
df1.to_excel('av.xlsx',sheet_name='sh')


#####排序
data.columns[1]
data1=data.sort_index(axis=1,ascending=True)

data1=data.sort_index()
data1=data.sort_values(by=['单位'])

import locale
sudo apt-get install language-pack-zh*
locale.setlocale('LC_COLLATE', 'zh_CN.UTF8')
a = ['中国人', '啊', '你好', '台湾人']
b = sorted(a, cmp = locale.strcoll)

##############汉字转拼音
import os
os.getcwd()
os.chdir("e:\\bank")   #修改当前工作目录
def convert(ch):
    """该函数通过输入汉字返回其拼音，如果输入多个汉字，则返回第一个汉字拼音.
       如果输入数字字符串，或者输入英文字母，则返回其本身(英文字母如果为大写，转化为小写)
    """
    length = len('柯') #测试汉字占用字节数，utf-8，汉字占用3字节.bg2312，汉字占用2字节
    intord = ord(ch[0:1])
    if (intord >= 48 and intord <= 57):
        return ch[0:1]
    if (intord >= 65 and intord <=90 ) or (intord >= 97 and intord <=122):
        return ch[0:1].lower()
    ch = ch[0:length] #多个汉字只获取第一个
    with open('e:\\bank\\convert-utf-8.txt',encoding='utf-8') as f:
        for line in f:
            if ch in line:
                print(line)
                a=line[length:len(line)-1]
convert('个')
ch='好'v
ord('地')

f = open('e:/python/data.csv', 'r')  # 文件为123.txt
sourceInLines = f.readlines()  # 按行读出文件内容
f.close()
###stack 多级index指引
data.index=data.单位
data.drop('单位',1,inplace=True)

data_=data.stack()
z=data_.index.values
z1=list(z)
z2=list(zip(*z))[1]


c = pd.DataFrame({'a':data_})
c['b']=z2
t3=c.a+1
t4=c.b.str.find('表')
t5=c.b+c.a.str()

len(data_)

t1=np.tile(data.单位,3)

def __clear_env():
    for key in globals().keys():
        if not key.startswith("__"):  # 排除系统内建函数
            #globals().pop(key)
            del key
for i in dir():
    if not i.startswith('__'):
        locals().pop(i)
a=dir()
for key in dir():
    if not key.startswith("__"):  # 排除系统内建函数
        # globals().pop(key)
        del key
