# coding = unicode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
s = pd.Series([1,3,5,np.nan,6,8])

dates = pd.date_range('20130101', periods=6,freq='d')
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A':1.,
'B':pd.Timestamp('20130102'),
'C':pd.Series(1, index=list(range(4)),dtype='float32'),
'D':np.array([3] * 4, dtype='int32'),
'E':pd.Categorical(['test','train', 'test','train']),
'F':'foo'
})
df.tail(2)
t=df.values
t[1]
t=df.A.values
df[0:3]
t=df.loc[dates[0]]
t=df.A

t=dates[0]
df.loc[[20130102,20130104],['A','B']]
t=df.loc['20130102':'20130104', ['A','B']]
df.loc['20130102', ['A','B']]
df.loc[dates[0:3], 'A']
df.at[dates[0], 'A']

df.iloc[1,:]

#布尔值
df[df.A>0]
df.loc[df.A>0,'A']
df>0
df2=df.copy()
df2['E']=['one', 'one', 'two', 'three', 'four', 'three']
df2.E
df2[df2['E'].isin(['two', 'four'])]
#合并，左连接
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
df['F']=s1
df.pop('F')
df['F']=2
np.array([5] * len(df))
df.iat[1,1]=np.nan
df.dropna()
df[pd.isnull(df)]=2

s = pd.Series([1,3,4,np.nan,6,8], index=dates).shift(3)

df.apply(np.cumsum)

t=np.linspace(1,2,10)
t.sum()
s=pd.Series(np.random.randint(0, 7, size=4),index=list('abcd'))
s.value_counts()
s[1]
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()
#合并
df=pd.DataFrame(np.random.randn(10,4))
df2[]
t1=df[:3]
t2=df[7:]
t3=t2.sort_index(axis=1,ascending=False)
pd.concat([t1,t3])

#左右连接
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1,2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left,right,on='key')

df=pd.DataFrame(np.random.randn(8,4),columns=list('ABCD'))

s=df.iloc[3]
df.append(s, ignore_index=True)

df = pd.DataFrame({
'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
'C': np.random.randn(8),
'D': np.random.randn(8)
})
df.groupby('A').sum().C
df.groupby(['A', 'B']).sum()

x = [1, 2, 3]
y = [4, 5, 6]
xyz = zip(x,y)
print(list(xyz))

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
'one', 'two', 'one', 'two']]))

tuples = list(zip(['bar', 'bar', 'baz', 'baz',
'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
'one', 'two', 'one', 'two']))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
stacked = df2.stack()

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.head()
t=(prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

t3=prng[0]
(t3.asfreq('M','e')+1).asfreq('H', 's') + 9
#分类
df = pd.DataFrame({
'id':[1,2,3,4,5,6],
'raw_grade':['a','b','b','a','a','e']
})
df['grade'] = df['raw_grade'].astype('category',ordered=True)
z=df.grade
prng.asfreq('M', 'e') + 1).asfreq('H', 's')+9
df['grade'].cat.categories = ['very good', 'good', 'very bad']
df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium', 'good', 'very good'])
df.groupby("grade").size()
ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df=df.cumsum()
plt.figure;df.plot();plt.legend(loc='best')

df.to_csv('foo.csv')
z=pd.read_csv('foo.csv')



a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)

t=tuples[1]



















u = u'汉'
print (repr(u)) # u'\u6c49'
s = u.encode('UTF-8')
print (repr(s)) # '\xe6\xb1\x89'
u2 = s.decode('UTF-8')
print repr(u2) # u'\u6c49'

import codecs
f = codecs.open("e:/python/tip.txt")
content = f.read()
f.close()
print (content)


t=dir()
del t
for i in dir():
    del i
nums=[1,2,3,4]
nums=np.arange(4)
lst = [1,2,3,4,5]
lstiter = iter(lst)
for i in range(len(lst)):
    print (lstiter.next())
    1
    2
    3
    4
    5
for key in globals().keys():
    if not key.startswith("__"):
        globals().pop(key)

a=globals().keys()

del a
t=globals().keys()
t{1}
globals().pop(5)

