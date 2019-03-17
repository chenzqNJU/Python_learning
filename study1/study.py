
(x, y, z) = (1, 2, 'a string')
x, y, z= 1, 2,\
'a string'


print(np.asmatrix.__doc__)4
type(readCSV)
print(readCSV)

np.int('2')
print(pow())

np.coerce(1,2)
ord('1')


['a','b']*2
('a','b')*3

t=list(range(4))

'ab' in 'abcd'

s = ';'.join(('Spanish', 'Inquisition', 'Made Easy'))
s.upper()

'dfdf '.rstrip()
unicode('得到')

t=list('abcd').pop(index=1)

d=dict(x=1, y=2)


it=iter(fr)
it.next()

# -*- coding:utf-8 -*-

f = open('e:/python/data.csv', 'r')  # 文件为123.txt
sourceInLines = f.readlines()  # 按行读出文件内容
f.close()
new = []  # 定义一个空列表，用来存储结果
for line in sourceInLines:
    temp1 = line.strip('\n')  # 去掉每行最后的换行符'\n'
    temp2 = temp1.split(',')  # 以','为标志，将每行分割成列表
    new.append(temp2)  # 将上一步得到的列表添加到new中

print
new

new = []
for q in fr:
    new.append(q)
fr.readline().strip()
fr.text()
fr.name
[line.strip() for line in fr.readlines()]
[line.strip() for line in fr]
fr = open('e:/python/data.csv', 'r')

fr.seek(0)
z=[word for line in fr for word in line.split(',')]

print(filter(lambda n: n%2, [1,2,3,4,5]))

from random import randint
allNums = []
for eachNum in range(9):
    allNums.append(randint(1, 99))
print (filter(lambda n: n%2, allNums))
list(filter(lambda n: n%2, allNums))

def odd(n):
    return n % 2
allNums = []
for eachNum in range(9):
    allNums.append(randint(1, 99))

list(map((lambda x: x+2), [0, 1, 2, 3, 4, 5]))



















