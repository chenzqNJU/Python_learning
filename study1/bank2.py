import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

z=[]
for each in result:
    if 'G03' in each:
        z.append(each)
length=len(z)

p1 = "数据表.*?农商"
pattern1 = re.compile(p1)#我们在编译这段正则表达式
z4=[int(re.sub("\D","",re.search(pattern1,key).group(0))) for key in z]

year=range(2014,2018)
z8=[]
for n,x in enumerate(z):
    temp=[x.count(str(i)) for i in year]
    if temp[1]>0 and temp[2]>0:
        print("error")
    if temp[1]>0 or temp[2]>0:
        temp[0],temp[3]=0,0
    elif temp[0]>temp[3]:
        temp[3]=0
    else:
        temp[0]=0
    z8.append(np.int(np.array(year)[np.array(temp)>0]))

#仅包含xls xlm文件
list(filter(lambda x:not x.__contains__('xls'),z))
i=350
shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        print(i)
        continue
    try:
        data= pd.read_excel(z[i])
    except:
        print("t",i)
        continue
    if not data.iloc[0,:].str.contains('报表日期').any():
        print("err",i)

    data
    shape=list(data.shape)
    shape1.append(shape)
t=data.iloc[:,2]
t1=t.str.contains('报表日期')
z[350]

final=pd.DataFrame()
for i in range(length):
    try:
        data= pd.read_excel(z[i])
    except:
        continue
    data1=data.iloc[:,:9]
    data1['n']=z4[i]
    data1['year']=z8[i]
    final=final.append(data1,ignore_index=True)

###########################################
z=[]
for each in result:
    if 'G01' in each:
        z.append(each)
length=len(z)


p1 = "G010[1-9].*"
pattern1 = re.compile(p1)#我们在编译这段正则表达式
z1=list(filter(lambda key:not re.search(pattern1,key),z))
p1 = "第.*?部分"
pattern1 = re.compile(p1)
z2=list(filter(lambda key:not re.search(pattern1,key),z1))
length=len(z2)

p1 = "数据表.*?农商"
pattern1 = re.compile(p1)#我们在编译这段正则表达式
z4=[int(re.sub("\D","",re.search(pattern1,key).group(0))) for key in z2]
year=range(2014,2018)
z8=[]
for n,x in enumerate(z2):
    temp=[x.count(str(i)) for i in year]
    if temp[1]>0 and temp[2]>0:
        print("error")
    if temp[1]>0 or temp[2]>0:
        temp[0],temp[3]=0,0
    elif temp[0]>temp[3]:
        temp[3]=0
    else:
        temp[0]=0
    if n==310:
        temp[2]=0
    z8.append(np.int(np.array(year)[np.array(temp)>0]))

z2[310]
shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z2[i])
    except:
        continue
    if data.columns[0].__contains__('附注'):
        print(z2[i])
    shape=list(data.shape)
    shape1.append(shape)

final=pd.DataFrame()
for i in range(length):
    try:
        data= pd.read_excel(z2[i])
    except:
        continue
    if data.columns[0].__contains__('附注'):
        continue
    data1=data.iloc[:,:5]
    data1['n']=z4[i]
    data1['year']=z8[i]
    data1['flag'] = i
    final=final.append(data1,ignore_index=True)

final.to_csv('e:\\bank\\G01.csv',encoding='GBK')

del data1
data = pd.read_excel(z2[43])
data.columns[0]

######################资本充足率
z=[]
for each in result:
    if 'G40' in each:
        z.append(each)
length=len(z)

z1=[x[30:] for x in z if not '12' in x]
z2=[x for x in z1]

p1 = "G010[1-9].*"
pattern1 = re.compile(p1)#我们在编译这段正则表达式
z1=list(filter(lambda key:not re.search(pattern1,key),z))
p1 = "第.*?部分"
pattern1 = re.compile(p1)
z2=list(filter(lambda key:not re.search(pattern1,key),z1))
length=len(z2)

p1 = "数据表.*?农商"
pattern1 = re.compile(p1)#我们在编译这段正则表达式
z4=[int(re.sub("\D","",re.search(pattern1,key).group(0))) for key in z2]
year=range(2014,2018)
z8=[]
for n,x in enumerate(z2):
    temp=[x.count(str(i)) for i in year]
    if temp[1]>0 and temp[2]>0:
        print("error")
    if temp[1]>0 or temp[2]>0:
        temp[0],temp[3]=0,0
    elif temp[0]>temp[3]:
        temp[3]=0
    else:
        temp[0]=0
    if n==310:
        temp[2]=0
    z8.append(np.int(np.array(year)[np.array(temp)>0]))

z2[310]
shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z2[i])
    except:
        continue
    if data.columns[0].__contains__('附注'):
        print(z2[i])
    shape=list(data.shape)
    shape1.append(shape)

final=pd.DataFrame()
for i in range(length):
    try:
        data= pd.read_excel(z2[i])
    except:
        continue
    if data.columns[0].__contains__('附注'):
        continue
    data1=data.iloc[:,:5]
    data1['n']=z4[i]
    data1['year']=z8[i]
    data1['flag'] = i
    final=final.append(data1,ignore_index=True)

final.to_csv('e:\\bank\\G01.csv',encoding='GBK')

############################################S6301

z=[]
for each in result:
    if 'S63' in each:
        z.append(each)

z1=[x for x in z if not ('S6302' in x or 'S6303' in x)]
z2=[x for x in z1 if not ('II' in x or 'xml' in x)]
def get_year(z):
    year=range(2014,2018)
    z8=[]
    for n,x in enumerate(z):
        temp=[x.count(str(i)) for i in year]
        if temp[1]>0 and temp[2]>0:
            print("error")
        temp[1]*=100;temp[2]*=100;
        z8+=[(year[np.argmax(temp)])]
    return z8
def get_num(z):
    p1 = "数据表.*?农商"
    pattern1 = re.compile(p1)  # 我们在编译这段正则表达式
    z4 = [int(re.sub("\D", "", re.search(pattern1, key).group(0))) for key in z2]
    return z4
def get_month(z3):
    month=[]
    pattern1 = re.compile("(?<=201[4-7])[0-9]{1,2}")
    pattern2 = re.compile('(?<=年)[\d]{1,2}(?=月)')
    for i in z3:
        i=i.replace('.','')
        if re.search(pattern1,i):
            month+=[re.search(pattern1,i).group(0)]
        elif re.search(pattern2, i):
            month += [re.search(pattern2, i).group(0)]
        else:month+=['12']
    return [int(x) for x in month]

df2 = pd.DataFrame({'num':get_num(z2),'year':get_year(z2),'month':get_month(z2)})
df2=df2.sort_values(['num','year','month'])
df3=df2.drop_duplicates(['num','year'],'last')
flag=list(df3.index)
z3=[x for i,x in enumerate(z2) if i in flag]
year=get_year(z3);num=get_num(z3);

data= pd.read_excel(z3[1])
shape1=[]
for i in range(length):
    if z[i][-3:] != 'xls':
        continue
    try:
        data= pd.read_excel(z3[i])
    except:
        continue
    shape1.append(shape)

final=pd.DataFrame()
for i in range(length):
    try:
        data= pd.read_excel(z3[i])
    except:
        continue
    data1=data.iloc[:,:10]
    data1['n']=num[i]
    data1['year']=year[i]
    data1['flag'] = i
    final=final.append(data1,ignore_index=True)
final.to_csv('e:\\bank\\S6301.csv',encoding='GBK')

final.iloc[:,1].fillna('-1',inplace=True)
final.set_index(['n','year'],inplace=True)
final.columns=final.iloc[3,:].values
final1=final[final.iloc[:,1].str.lstrip().str.startswith('5')]

name=[re.sub('\d','',x) for x in final1['-1']]
name1=[x.replace('.','').strip() for x in name]
final1.loc[:,['-1']]=name1


final2=pd.pivot_table(final1,index=["-1"],values=final1.columns[2:-1],
               columns=["year"],aggfunc=[np.sum],fill_value=0)
#修改1级列名
a=list(zip(*(final2.columns)))
a[0]=tuple(np.array([6,2,7,8,5,1,3,4]).repeat(4))
final2.columns=pd.MultiIndex.from_tuples(list(zip(*a)))

final2_=pd.pivot_table(final1,index=["-1"],values=final1.columns[2:-1],
               columns=["year"],aggfunc=[len])
final2_.columns=final2.columns
final3=final2/final2_

a=[[4,0,1,3,2]]+[list(final3.index)]
final3.index=pd.MultiIndex.from_tuples(list(zip(*a)))
final4=final3.sort_index(level=0).sort_index(axis=1,level=[0,2])


final4.to_excel('a.xlsx')

final3.columns.names[1]=['a']









#找出一年多个报表的
x=get_num(z2)
z3=[z2[x1] for x1,x2 in enumerate(x) if x.count(x2)>4]
#删除2014年11月类似的，其实也不对，有些没有12月以11月替代
pattern1 = re.compile('(?<=年).{1,2}(?=月)')
z4=list(filter(lambda key:not (re.search(pattern1,key) and int(re.search(pattern1,key).group(0))<12),z3))
#找出月份
p1 = "(?<=201[4-7])[0-9]{1,2}"
pattern1 = re.compile("(?<=201[4-7])[0-9]{1,2}")
re.search(pattern1,z3[12]).group(0)
pattern2 = re.compile('(?<=年).{1,2}(?=月)')



z3_=[]
for i in range(len(z3)):
    if not (re.search(pattern1,z3[i]) or re.search(pattern2, z3[i])):
        z3_+=[z3[i]]



re.sub("\D", "",z3[1]).replace('20141104','')
p1='(?<=年).{1,2}(?=月)'
pattern1 = re.compile(p1)
z4=list(filter(lambda key:not (re.search(pattern1,key) and int(re.search(pattern1,key).group(0))<12),z3))




int('01')
myset = np.array(list(set(x)))
cishu=np.array([x.count(t) for t in myset])
cishu1=myset[cishu>4]
cishu2=[t1 for t1,t2 in enumerate(x) if t2 in cishu1]
z2[cishu2]

help(sorted)
list(x.index)


l1 = ['b','c','d','b','c','a','a']
l2 = sorted(set(l1),key=l1.index)

x=get_num(z2)
x=z2{cishu2}

############################################################

key=z[36]
a=re.search(pattern1,key)
a==None
z4=[int(re.sub("\D","",re.search(pattern1,key).group(0))) for key in z]

z1[5].find('VIII')

##################################苏北苏中苏南
import gc

os.chdir("e:\\bank")
data1=pd.read_excel('62家农商行地区分布(1).xlsx',d=None)
data1=data1.dropna(how='all')
data1.b=data1.b.str.replace('（','').str.replace('）','')   #删除“（南京）”的括号
#####填充缺失值
for i,x in enumerate(data1.a):
    if pd.isnull(x):
        data1.iloc[i,0]=data1.iloc[i-1,0]
data1.index=[data1.a,data1.b]

data1=data1.drop(['a','b'],axis=1)
data2=data1.stack()
z=list(data2.index.values)
z1=list(zip(*z))[0];z2=list(zip(*z))[1]
data2.index=[list(z1),list(z2)]

fenlei=data2.to_csv('fenlei.csv',encoding='GBK')






'（地 方）'.translate('（）','  ')
'（地 方）'.replace('（）','')


[x if x!=np.nan else  for x in a]
help(pd.read_excel)








final.to_csv('e:\\bank\\G03.csv',encoding='GBK')







list(filter(lambda x:not x.__contains__('xls'),z))







year[np.array(temp)>0]
bool(temp>0)
list(np.array(year)[np.array(temp)>0])
x=z[1]
key=z[1]
pattern1 = re.compile(p1)#我们在编译这段正则表达式
matcher1 = re.sub("\D","",re.search(pattern1,key).group(0))

z4=[int(re.sub("\D","",re.search(pattern1,key).group(0))) for key in z]
re.search(pattern1,z3[1])

key=z3[12]
a=lambda f:re.search(pattern1,key) and re.search(pattern1,key).group(0)

key=z3[14]
bool(re.search(pattern1,key) and int(re.search(pattern1,key).group(0))==12)

z4=list(filter(lambda key:not (re.search(pattern1,key) and int(re.search(pattern1,key).group(0))<12),z3))


f=lambda key:re.search(pattern1,key) and int(re.search(pattern1,key).group(0))==12
for i in range(len(z3)):
    f(z3[i])

reduce(lambda f,i:f.append((f[-2]+f[-1])) or f,range(10),[1,1])

int(matcher1)


totalCount = '100'
totalPage = int(totalCount)/20

z4=[]
i=16
for i in range(len(z)):
    key=z[i]
    matcher1=re.search(pattern1,key).group(0)

key=r'数据表_10农商行'