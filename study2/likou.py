import myfunc
import numpy as np
import pandas as pd
help(np.concatenate)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b),axis=0)
np.concatenate((a, b.T),axis=1)
np.append(a,b)
np.append(a,b,axis=0)
np.append(a,b.T,axis=1)
np.row_stack((a,b))
np.column_stack((a,b.T))
np.repeat(a,2,axis=1)
np.tile(a,(2,2))
help(np.tile)
help(pd.DataFrame.join)
help(pd.DataFrame.append)

df1 = pd.DataFrame({'key': ['a', 'b', 'b'], 'data1': range(3)})
df2 = pd.DataFrame({'key': ['a', 'b', 'c'], 'data2': range(3)})
pd.merge(df1, df2,on='key',how='outer')        #默认以重叠的列名当做连接键。默认how=inner连接
df1 = pd.DataFrame({'key1': ['a', 'b', 'b'], 'data1': range(3)})
df2 = pd.DataFrame({'key2': ['a', 'b', 'c'], 'data2': range(3)})
pd.merge(df1, df2,left_on='key1',right_on='key2',how='outer')
pd.merge(df1, df2,left_index=True,right_index=True,how='outer',suffixes=('_1', '_2'))
df1.join(df2,lsuffix='_1', rsuffix='_2') ##按index合并出现相同的列名，添加后缀

pd.concat([df1,df2],axis=1)                         # outer合并，有相同的列key，打开会卡
pd.concat([df1,df2],axis=1,keys=['a','b'])          # 再分配一层索引
pd.concat([df1,df2],axis=1,join_axes=[df1.index])   # 根据df1来合并，类似于左连接
df1 = pd.DataFrame({'key': ['a', 'b'], 'data1': range(2)}).set_index('key')
df2 = pd.DataFrame({'key': ['a', 'c'], 'data2': range(2)}).set_index('key')
pd.concat([df1,df2,df2],axis=1,join_axes=[df1.index],sort=False)

import queue
import collections
a=queue.Queue()
a.queue=[1,4,6,7,9,0,4,6]
a.put(2)
a.get()

a=queue.PriorityQueue()
a.queue=collections.deque([1,4,6,7,9,0,4,6])
a.get()

for i in [1,4,6,7,9,0,4,6]:
    a.put(i)
    print(a.queue)
a.queue.pop(5)
a.get()
a.put(7)
a.pop()

c=collections.deque([1,4,6,7,9,0,4,6])
c.pop()
c.rotate(-1)



nums=[1,4,6,7,9,0,4,6]
nums.sort()

nums.insert(0,0)
nums.pop(1)



def rob(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # 不同情况判断：空数组，数组元素1个，数组元素2个
    l = len(nums)
    if l == 0:
        return 0
    elif l == 1:
        return nums[0]
    elif l == 2:
        return max(nums[0], nums[1])

    # 构建opt数组
    opt = [0] * len(nums)
    opt[0] = nums[0]
    opt[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        A = opt[i - 2] + nums[i]
        B = opt[i - 1]
        opt[i] = max(A, B)

    return opt[-1]

opt = [0] * len(nums)
opt[0] = nums[0]
opt[1] = max(nums[0], nums[1])

def rob(nums):
    if len(nums)<=3:return nums[-1]
    # if len(nums) == 2: return nums[0]
    A = rob(nums[:-2]) + nums[-1]
    B = rob(nums[:-1])
    return max(A, B)


def rob(nums):
    if len(nums) == 1:return nums[-1]
    if len(nums) == 2: return max(nums)
    A = rob(nums[:-2]) + nums[-1]
    B = rob(nums[:-1])
    return max(A, B)

max([1,2])

nums=[1,2,3,9,6]
rob(nums)

nums=[0]
rob(nums)


def rob(self, nums: 'List[int]') -> 'int':
    last = 0
    now = 0
    for i in nums:
        last, now = now, max(last + i, now)
    return now

nums=[[1,3,1],[1,5,1],[4,2,1]]
def minPathSum(nums):

    m = len(nums)
    n = len(nums[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                nums[i][j] += nums[i][j-1]
            elif j == 0:
                nums[i][j] += nums[i-1][j]
            else:
                nums[i][j] += min(nums[i-1][j],nums[i][j-1])
    nums[i][j]

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

a=ListNode([1,2,3,54])
a.val
a.next

import myfunc
myfunc.search('sort_values')
import pandas as pd
a=pd.DataFrame([[1,2],[1,3]],columns=list('ab'))

a.sort_values('a').drop_duplicates('a')
[1,2,3,1,2,3,5].count(5)

import numpy as np
a=np.matrix([[1,2],[3,4]])
a*a.T

X=np.array([1,2,3,4,5])
X=X.reshape(-1, 1)
Y=np.array([0,2,3,4,6])
Y=Y.reshape(-1, 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)
model.coef_

np.cov(X.T,Y.T)
np.var(X.T)

a=np.concatenate((X,Y),axis=1)
np.cov(a.T)


X=np.array([[1 ,5 ,6] ,[4 ,3 ,9 ],[ 4 ,2 ,9],[ 4 ,7 ,2]])
x=X[0:2]


meanX, meanY = np.mean(X), np.mean(Y)
n = np.shape(X)[0]
#按照协方差公式计算协方差，Note:分母一定是n-1
covariance = sum(np.multiply(X-meanX, Y-meanY))/(n)

np.corrcoef(X.T,Y.T)


list(1)


nums=[-2,1,-3,4,-1,2,1,-5,4]
def maxSubArray(self, nums):
    max_num = [nums[0]]
    for n in nums[1:]:
        if max_num[-1] > 0:
            max_num.append(max_num[-1]+n)
        else:
            max_num.append(n)
    return max(max_num)

def maxSubArray(self, nums):
    now = 0;last = 0;t=[]
    for i in nums:
        last, now = now, max(last , 0)+i

        last = now
        now = max(last , 0)+i

        print(now)

    return now

    last, now = now,last+1

    last = 0
    now = 0
    for i in nums:
        last, now = now, max(last + i, now)
    return now



8&7


int(-1.9)

3/2==3//2
n=23
for i in range(2,int(np.sqrt(n))+1):
    if n/i==n//i:
        print(1)


a = [0]*n
for i in range(n):
    if i%2==0:a[i]=1


for i in range(2,int(np.sqrt(n))+1,2):
    if a[i]==1:


l = list(range(0, n + 1))
l[1] = 0
for i in range(2, n + 1):
    if l[i]:
        for j in range(i*2,n+1,i):
            l[j]=0


def getSum(a, b):
    if(b == 0):return a;
    else:
        return getSum(a ^ b, (a & b) << 1)


getSum(4,5)


4^5
(4 & 5) << 1
4<<1

1^1


sorted([1,2,3,1])
list(reversed([1,2,3]))

a=[1,2,3,4,5,6]
a[-1:0:-1]


a.append(2)
a.extend([2])

str(bin(n))[2:]

bin(7)

oct(-8)

hex(16)

int('10',16)

n=6
count = 0
while n:
    count = count + 1
    n = n & (n - 1)

bin(n-1)

nums=[1,2,3,4,5,6,7,8];target=5
l, r = 0, len(nums) - 1
while l <= r:
    mid = (l + r) // 2
    if nums[mid] > target:
        r = mid - 1
    elif nums[mid] < target:
        l = mid + 1
    else:
        print(mid);break


result = ''
alpha = '0123456789abcdef'
num=17

while num != 0 and len(result) < 8:
    result += alpha[num & 0xf]
    num = num >> 4
result[::-1]

2&1

bool(~bool(0))

bin(num)

oct(13)

0b1
bin(0xf)
~1
bin(12)
bin(~12)
bin(-6)

import numpy as np
~np.array([True,False,True,True,False])

~np.array([True])

##################################################
######################################## 冒泡排序
lyst=[1,4,5,6,2,3]
def bubbleSort(lyst):
    n = len(lyst)
    while n>1:
        i = 1
        while i<n:
            if lyst[i] < lyst[i-1]:
                lyst[i], lyst[i-1] = lyst[i-1], lyst[i]
            i +=1
        n -= 1

lyst[1],lyst[2]=lyst[2],lyst[1]


##################################################### 希尔排序
c=[7,10,0,12,1,5,3,11,4,9,6,8,2,13]
for D in [5,3,1]:#[5,3,1]是一个增量序列，数据量少时可以自己定义，尽量使元素间互质；数据量多时可以参考Sedgewick增量序列等。
    for p in range(D,len(c)):
        Tmp=c[p]
        for i in range(p,D-1,-D):
            if c[i-D]>Tmp:
                c[i],c[i-D]=c[i-D],c[i]
            else:break



nums=[2, 7, 11, 15];target=9
l, r = 0, len(nums) - 1
while l <= r:
    if nums[l]+nums[r]>9:
        r-=1
    elif nums[l]+nums[r]<9:
        r+=1
    else:
        break

########################################################## 快速排序
def QuickSort(myList,start,end):
    if start < end:
        i,j = start,end
        base = myList[i]

        while i < j:
            while (i < j) and (myList[j] >= base):
                j = j - 1
            myList[i] = myList[j]
            while (i < j) and (myList[i] <= base):
                i = i + 1
            myList[j] = myList[i]
        myList[i] = base

        QuickSort(myList, start, i - 1)
        QuickSort(myList, j + 1, end)
    return myList
myList = [49,38,65,97,76,13,27,49]
QuickSort(myList,0,len(myList)-1)

del QuickSort
def QuickSort(myList):
    if len(myList)<2:return myList
    L=QuickSort(list(filter(lambda x: x <= myList[0], myList[1:])))
    R=QuickSort(list(filter(lambda x: x >= myList[0], myList[1:])))
    myList=L+[myList[0]]+R
    return myList
myList=[1,2,0]
QuickSort(myList)

############################################################ 堆排序
L = [50, 16, 30, 10, 60,  90,  2, 80, 70]
L.insert(0,0)

start, end=2,9
def heap_adjust(L, start, end):
    temp = L[start]
    i = start
    j = 2 * i
    while j <= end:
        if (j < end) and (L[j] < L[j + 1]):
            j += 1
        if temp < L[j]:
            L[i] = L[j]
            i = j
            j = 2 * i
        else:
            break
    L[i] = temp

#####################################siftDown siftUp 算法
def heap_(L,start,end):
    node=2*start
    if node>end:return L
    if node<end and L[node]<L[node+1]:node+=1
    if L[node]>L[start]:L[node],L[start]=L[start],L[node]
    heap_(L, node, end)
    return L

def heap_sort(L):
    L_length = len(L) - 1
    first_sort_count = L_length >> 1
    for i in range(first_sort_count):
        heap_(L, first_sort_count - i, L_length)

    for i in range(L_length - 1):
        L[1],L[L_length - i]=L_length - i,L[1]
        heap_(L, 1, L_length - i - 1)

    print(L)



import numpy as np
data = np.array([[1,5,3],[4,5,6]])
data.sort()

data = np.array([1,5,3])
data.argsort()


for i in range(1,10):
    print('\n')
    for j in range(1,i+1):
        print('%d*%d=%-2d' % (i, j, i * j),end=' ')

x=3
print(' '.join(["fizz"[x % 3 * 4:]+"buzz"[x % 5 * 4:] or str(x) for x in range(1, 101)]))


print('\n'.join([' '.join(['%s*%s=%-2s' % (y, x, x*y) for y in range(1, x+1)]) for x in range(1, 10)]))


'%s*%s=%-2s' % (1, 2, 3)

print ("His name is %s"%"Aviad")

print ("His height is %f m"%1.83)

("His height is {} m").format(2)

"His name is %06d"%230


print("%+10x" % 10)
print("%04d" % 5)
print("%6.3f" % 2.3)

filter(lambda x: not [x % i for i in range(2, x) if x % i == 0], range(2, 101))
filter(lambda x: all(map(lambda p: x % p != 0, range(2, x))), range(2, 101))


x=23
not [x for i in range(2, x) if x % i == 0]
all(map(lambda p: x % p != 0, range(2, x)))
list(map(lambda p: x % p != 0, range(2, x)))

del a
[(a,a[i][0], a.append([a[i][1], a[i][0]+a[i][1]])) for a in ([[1, 1]], ) for i in range(5)]

[a.append([a[i][1], a[i][0]+a[i][1]]) for a in ([[1, 1]], ) for i in range(5)]

a =[[1, 1]]
for i in range(5):
    a.append([a[i][1], a[i][0] + a[i][1]])

[a for a in ([[1, 1]])]

a=[1]
a.append([1,2])


###########  快排
qsort = lambda arr: \
    len(arr) > 1 and qsort(list(filter(lambda x: x <= arr[0], arr[1:]))) + arr[0:1] \
    + qsort(list(filter(lambda x: x > arr[0], arr[1:]))) or arr


qsort([1,3,2])

arr=[1,2]

import itertools
list(itertools.compress([1,2,3,4,5],[1,0,1,1,1]))

x,y=[1,2,3,4,5],[1,0,1,1,1]
[d for d, s in zip(x, y) if s]

help(itertools.dropwhile)

from itertools import *

for i in zip(count(1), ['a', 'b', 'c']):
    print(i)
a=count(1)
for i in zip(cycle([1,2]), ['a', 'b', 'c']):
    print(i)

for i in repeat('a', 5):
    print(i)
a=repeat('a', 5)

list(chain([1, 2, 3], ['a', 'b', 'c']))
list(chain.from_iterable([[1,2,3],'abc']))
list(filterfalse(lambda x:x%2,range(5)))
list(filter(lambda x:~x%2,range(5)))


print(a.__next__())

a.__next__()

list(dropwhile(lambda x:x<3,[1,2,3,4,1,6]))
list(takewhile(lambda x:x<3,[1,2,3,4,1,6]))

a = ['aa', 'ab', 'abc', 'bcd', 'abcde']
for i, k in groupby(a, len):
    print(i, list(k))

from operator import itemgetter
d = dict(a=1, b=2, c=1, d=2, e=1, f=2, g=3)
di = sorted(d.items(), key=itemgetter(1))

for k, g in groupby(di, key=itemgetter(1)):
    print(k, list(map(itemgetter(0), g)))

for i in islice(count(5), 5):
    print(i)

values = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
list(starmap(lambda x,y:(x, y, x*y), values))

list(map(lambda x,y:(x, y, x*y), (0, 1, 2, 3, 4),(5, 6, 7, 8, 9)))
list(zip(*values))

r = islice(count(), 5)
i1, i2 = tee(r)

list(zip_longest('abc','AB',fillvalue=None))

# 笛卡尔
a = (1, 2, 3)
b = ('A', 'B', 'C')
list(product(a,b))
list(product([1,2,3],repeat=2))
# 排列
list(permutations([1,2,3],2))
# 组合
list(combinations([1,2,3],2))
list(combinations_with_replacement([1,2,3],2))

import myfunc
myfunc.search('reduce')


from functools import reduce
reduce(lambda f,i:f.append((f[-2]+f[-1])) or f,range(10),[1,1])

print(reduce(lambda x, y: x+y, range(10)))    # 45
print(reduce(lambda x, y: x+y, range(10), 100))    # 145
print(reduce(lambda x, y: x+y, [[1, 2], [3, 4]], [0]))    # [0, 1, 2, 3, 4]


a=[[1,1]]
for i in range(5):
    a.append([a[i][1],a[i][0]+a[i][1]])

a=[1,1]
for i in range(5):
    a.append(a[-1]+a[-2])

9 or [1,2]

# kdj
a=[1,2,3,4,5,6,7]
print(reduce(lambda f,i:f.append(f[-1]/2+i/2) or f,a,[0]))

a=[1,1,1,1,4,4,4,1,1,2,2,2,3,3,3]
b=[]
for _, k in groupby(a):
    b.extend(range(list(k).__len__()))


19 & 1


0x11

a=np.matrix([[0,1],[1,1]])
b=np.matrix([1,1])

a*b.T
a**6*b.T
a**30

a*a
np.dot(a,a)
np.multiply(a,a)