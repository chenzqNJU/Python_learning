#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools

t=pd.DataFrame(pso.pg.reshape(m,D))
t.to_csv('e:\\jm\\temp2.csv')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 100)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return sum(individual),


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)


# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

help(np.linalg.norm)
# ----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)
    # len(pop[1])
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    list(zip(pop, fitnesses))
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()

import numpy as np

coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                        [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                        [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                        [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                        [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                        [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
                        [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
                        [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
                        [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
                        [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
                        [1340.0,725.0],[1740.0,245.0]])

def getdistmat(coordinates):
    num = coordinates.shape[0]
    distmat = np.zeros((52,52))
    for i in range(num):
        for j in range(i,num):
            distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])
    return distmat


def initpara():
    alpha = 0.99
    t = (1,100)
    markovlen = 10000

    return alpha,t,markovlen
num = coordinates.shape[0]
distmat = getdistmat(coordinates)

solutionnew = np.arange(num)
valuenew = np.inf

solutioncurrent = solutionnew.copy()
valuecurrent = np.inf

solutionbest = solutionnew.copy()
valuebest = np.inf

alpha,t2,markovlen = initpara()
t = t2[1]

result = [] #记录迭代过程中的最优解

while t > t2[0]:
    for i in np.arange(markovlen):

        #下面的两交换和三角换是两种扰动方式，用于产生新解
        if np.random.rand() > 0.5:# 两交换
            # np.random.rand()产生[0, 1)区间的均匀随机数
            while True:#产生两个不同的随机数
                loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                if loc1 != loc2:
                    break
            solutionnew[loc1],solutionnew[loc2] = solutionnew[loc2],solutionnew[loc1]
        else: #三交换
            while True:
                loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc3 = np.int(np.ceil(np.random.rand()*(num-1)))

                if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                    break

            # 下面的三个判断语句使得loc1<loc2<loc3
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1
            if loc2 > loc3:
                loc2,loc3 = loc3,loc2
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1

            #下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
            tmplist = solutionnew[loc1:loc2].copy()
            solutionnew[loc1:loc3-loc2+1+loc1] = solutionnew[loc2:loc3+1].copy()
            solutionnew[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()

        valuenew = 0
        for i in range(num-1):
            valuenew += distmat[solutionnew[i]][solutionnew[i+1]]
        valuenew += distmat[solutionnew[0]][solutionnew[51]]

        if valuenew < valuecurrent: #接受该解
            #更新solutioncurrent 和solutionbest
            valuecurrent = valuenew
            solutioncurrent = solutionnew.copy()

            if valuenew < valuebest:
                valuebest = valuenew
                solutionbest = solutionnew.copy()
        else:#按一定的概率接受该解
            if np.random.rand() < np.exp(-(valuenew-valuecurrent)/t):
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()
            else:
                solutionnew = solutioncurrent.copy()

    t = alpha*t
    result.append(valuebest)
    print(t) #程序运行时间较长，打印t来监视程序进展速度

import matplotlib.pyplot as plt
plt.plot(np.array(result))
plt.ylabel("bestvalue")
plt.xlabel("t")
plt.show()

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def inputfun(x):
    return (x-2)*(x+3)*(x+8)*(x-9)

initT = 1000 #初始温度
minT = 1 #温度下限
iterL = 1000 #每个T值的迭代次数
delta = 0.95 #温度衰减系数
k = 1

initx = 10*(2*np.random.rand()-1)
nowt = initT
print("初始解：",initx)

xx = np.linspace(-10,10,300)
yy = inputfun(xx)
plt.figure()
plt.plot(xx,yy)
plt.plot(initx,inputfun(initx),'o')
plt.show()

#模拟退火算法寻找最小值过程
while nowt>minT:
    for i in np.arange(1,iterL*10,1):
        funVal = inputfun(initx)
        xnew = initx+(2*np.random.rand()-1)
        if xnew>=-10 and xnew<=10:
            funnew = inputfun(xnew)
            res = funnew-funVal
            if res<0:
                initx = xnew
            else:
                p = np.exp(-(res)/(k*nowt))
                if np.random.rand()<p:
                    initx = xnew
#            print initx-xnew
#    print initx
#    print nowt
    nowt = nowt*delta

print("最优解：",initx)
print("最优值：",inputfun(initx))
plt.plot(initx,inputfun(initx),'*r')
plt.show()


############################求导
from numpy import poly1d
p = poly1d([3,4,5])         #多项式
print(p)

print(p * p)

print(p.integ(k=6))       # 积分

print(p.deriv())   #求导

p([4, 5])          #带入4,5后的值



from sympy import Symbol,solve,integrate,sin,pi,diff,Function,dsolve

x = Symbol('x')
y = Symbol('y')
# x,y=Symbol('x y')
print(solve([2*x-y-3,3*x+y-7],[x,y]))


t = Symbol('t')
x = Symbol('x')
m = integrate(sin(t)/(pi-t),(t,0,x))
n = integrate(m,(x,0,pi))

diff(x**3,x)
3*x**2
diff(x**3,x,2)
6*x

# 左端
f = Function('f')
x = Symbol('x')

diff(f(x), x)
# 看一下

f=2*x
print(diff(f(x), x))


f = Function('f')
x = Symbol('x')
print(2*x-diff(f(x),x))

dsolve(2*x - diff(f(x),x), f(x))
#result
#Eq(f(x), C1 + x**2)


############################################蚁群算法
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 15:21:03 2016
@author: SYSTEM
"""
import os
os.getcwd()
import numpy as np
import matplotlib.pyplot as plt

coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                        [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                        [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                        [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                        [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                        [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                        [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                        [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                        [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
                        [1340.0, 725.0], [1740.0, 245.0]])
a=list(zip(*coordinates))
a[1]
plt.scatter(a[0], a[1])
plt.show()
def getdistmat(coordinates):
    num = coordinates.shape[0]
    distmat = np.zeros((52, 52))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distmat


distmat = getdistmat(coordinates)

numant = 40  # 蚂蚁个数
numcity = coordinates.shape[0]  # 城市个数
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.1  # 信息素的挥发速度
Q = 1

iter = 0
itermax = 250

etatable = 1.0 / (distmat + np.diag([1e10] * numcity))  # 启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
pheromonetable = np.ones((numcity, numcity))  # 信息素矩阵
pathtable = np.zeros((numant, numcity)).astype(int)  # 路径记录表

distmat = getdistmat(coordinates)  # 城市的距离矩阵

lengthaver = np.zeros(itermax)  # 各代路径的平均长度
lengthbest = np.zeros(itermax)  # 各代及其之前遇到的最佳路径长度
pathbest = np.zeros((itermax, numcity))  # 各代及其之前遇到的最佳路径长度

while iter < itermax:

    # 随机产生各个蚂蚁的起点城市
    if numant <= numcity:  # 城市数比蚂蚁数多  np.random.permutation随机变换
        pathtable[:, 0] = np.random.permutation(range(0, numcity))[:numant]
    else:  # 蚂蚁数比城市数多，需要补足
        pathtable[:numcity, 0] = np.random.permutation(range(0, numcity))[:]
        pathtable[numcity:, 0] = np.random.permutation(range(0, numcity))[:numant - numcity]

    length = np.zeros(numant)  # 计算各个蚂蚁的路径距离

    for i in range(numant):

        # i=0
        visiting = pathtable[i, 0]  # 当前所在的城市

        # visited = set() #已访问过的城市，防止重复
        # visited.add(visiting) #增加元素
        unvisited = set(range(numcity))  # 未访问的城市
        unvisited.remove(visiting)  # 删除元素

        for j in range(1, numcity):  # 循环numcity-1次，访问剩余的numcity-1个城市
            # j=1
            # 每次用轮盘法选择下一个要访问的城市
            listunvisited = list(unvisited)

            probtrans = np.zeros(len(listunvisited))

            for k in range(len(listunvisited)):
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                               * np.power(etatable[visiting][listunvisited[k]], beta)
            cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()

            cumsumprobtrans -= np.random.rand()

            k = listunvisited[np.where(cumsumprobtrans > 0)[0][0]]  # 下一个要访问的城市

            pathtable[i, j] = k

            unvisited.remove(k)
            # visited.add(k)

            length[i] += distmat[visiting][k]

            visiting = k

        length[i] += distmat[visiting][pathtable[i, 0]]  # 蚂蚁的路径距离包括最后一个城市和第一个城市的距离

    # print length
    # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数

    lengthaver[iter] = length.mean()

    if iter == 0:
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy()
    else:
        if length.min() > lengthbest[iter - 1]:
            lengthbest[iter] = lengthbest[iter - 1]
            pathbest[iter] = pathbest[iter - 1].copy()

        else:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()

            # 更新信息素
    changepheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):
        for j in range(numcity - 1):
            changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                pathtable[i, j + 1]]

        changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]

    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

    iter += 1  # 迭代次数指示器+1

    # 观察程序执行进度，该功能是非必须的
    if (iter - 1) % 20 == 0:
        print(iter - 1)

# 做出平均路径长度和最优路径长度
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
axes[0].plot(lengthaver, 'k', marker=u'')
axes[0].set_title('Average Length')
axes[0].set_xlabel(u'iteration')

axes[1].plot(lengthbest, 'k', marker=u'')
axes[1].set_title('Best Length')
axes[1].set_xlabel(u'iteration')
plt.show()

fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
plt.close()

# 作出找到的最优路径图
bestpath = pathbest[-1]

plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker=u'$\cdot$')
plt.xlim([-100, 2000])
plt.ylim([-100, 1500])
bestpath=bestpath.astype(int)
for i in range(numcity - 1):  #
    m, n = int(bestpath[i]), int(bestpath[i + 1])
    print(m, n)
    plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')
plt.plot([coordinates[bestpath[0]][0], coordinates[n][0]], [coordinates[bestpath[0]][1], coordinates[n][1]], 'b')

ax = plt.gca()
ax.set_title("Best Path")
ax.set_xlabel('X axis')
ax.set_ylabel('Y_axis')
plt.show()

plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
plt.close()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pic(X, Y, Z, z_max, title, z_min=0):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    # ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    # plt.savefig("./myProject/Algorithm/pic/%s.png" % title) # 保存图片
    plt.show()


###################################################
# -*- coding: utf-8 -*-
"""
粒子群算法求解函数最大值（最小值）
f(x)= x + 10*sin5x + 7*cos4x
"""
import numpy as np
import matplotlib.pyplot as plt


# 粒子（鸟）
class Particle:
    def __init__(self):
        self.p = 0  # 粒子当前位置
        self.v = 0  # 粒子当前速度
        self.pbest = 0  # 粒子历史最好位置


class PSO:
    def __init__(self, N=20, iter_N=100):
        self.w = 0.2  # 惯性因子
        self.c1 = 1  # 自我认知学习因子
        self.c2 = 2  # 社会认知学习因子
        self.gbest = 0  # 种群当前最好位置
        self.N = N  # 种群中粒子数量
        self.POP = []  # 种群
        self.iter_N = iter_N  # 迭代次数

    # 适应度值计算函数
    def fitness(self, x):
        return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

    # 找到全局最优解
    def g_best(self, pop):
        for bird in pop:
            if bird.fitness > self.fitness(self.gbest):
                self.gbest = bird.p

    # 初始化种群
    def initPopulation(self, pop, N):
        for i in range(N):
            bird = Particle()
            bird.p = np.random.uniform(-10, 10)
            bird.fitness = self.fitness(bird.p)
            bird.pbest = bird.fitness
            pop.append(bird)

        # 找到种群中的最优位置
        self.g_best(pop)

    # 更新速度和位置
    def update(self, pop):
        for bird in pop:
            v = self.w * bird.v + self.c1 * np.random.random() * (
                    bird.pbest - bird.p) + self.c2 * np.random.random() * (self.gbest - bird.p)

            p = bird.p + v

            if -10 < p < 10:
                bird.p = p
                bird.v = v
                # 更新适应度
                bird.fitness = self.fitness(bird.p)

                # 是否需要更新本粒子历史最好位置
                if bird.fitness > self.fitness(bird.pbest):
                    bird.pbest = bird.p

    def implement(self):
        # 初始化种群
        self.initPopulation(self.POP, self.N)

        def func(x):
            return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

        x = np.linspace(-10, 10, 1000)
        y = func(x)

        # 迭代
        for i in range(self.iter_N):
            # 更新速度和位置
            self.update(self.POP)

            # 更新种群中最好位置
            self.g_best(self.POP)

            # 绘制动画
            plt.clf()
            scatter_x = np.array([ind.p for ind in pso.POP])
            scatter_y = np.array([ind.fitness for ind in pso.POP])

            scatter_x1 = pso.gbest
            scatter_y1 = pso.fitness(pso.gbest)

            plt.plot(x, y)
            plt.scatter(scatter_x, scatter_y, c='b')
            plt.scatter(scatter_x1, scatter_y1, c='r')
            plt.pause(0.01)


plt.close('all')
pso = PSO(N=20, iter_N=50)
pso.implement()

for ind in pso.POP:
    print("x = ", ind.p, "f(x) = ", ind.fitness)

print("最优解 x = ", pso.gbest, "相应最大值 f(x) = ", pso.fitness(pso.gbest))

plt.show()


#########################################
# -*- coding: utf-8 -*-
"""
f(x1,x2) = x1**2 + x2**2, x1,x2 belongs to [-10,10],求Minf
"""

import matplotlib.pyplot as plt
import numpy as np


class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 2  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-10, 10]  # 解空间范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # 初始化粒子群位置
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度

    def calculate_fitness(self, x):
        return np.sum(np.square(x), axis=1)

    def evolve(self):
        fig = plt.figure()
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.x = self.v + self.x
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.x_bound[0], self.x_bound[1])
            plt.pause(0.01)

            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)


#            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))


pso = PSO(10, 100)
pso.evolve()
plt.show()

##################################################som
import numpy as np
import pylab as pl


class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  形状是N*D， 输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        W=np.random.rand(X.shape[1], 5 * 5)
        print(self.W.shape)

    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a * b

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)
        N=3
        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            batch_size = 30
            X[np.random.choice(X.shape[0], batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        print(winner)
        return winner


def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X


def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W


# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()


if __name__ == '__main__':

    # 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

    a = data.split(',')
    dataset = np.mat([[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)])
    dataset_old = dataset.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    X=dataset
    som.train()
    res = som.train_result()
    classify = {}
    for i, win in enumerate(res):
        if not classify.get(win[0]):
            classify.setdefault(win[0], [i])
        else:
            classify[win[0]].append(i)
    C = []  # 未归一化的数据分类结果
    D = []  # 归一化的数据分类结果
    for i in classify.values():
        C.append(dataset_old[i].tolist())
        D.append(dataset[i].tolist())
    draw(C)
    draw(D)

help(np.random.choice)

np.random.choice(30,30)

###############################################
import numpy as np
import pylab as pl

output=(5,5)
class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  形状是N*D， 输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        print(self.W.shape)


    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a * b

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            train_X = X[np.random.choice(X.shape[0], batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            train_X.dot(W)

            winner = np.argmax(train_Y, axis=1).tolist()
            len(winner)
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        print(winner)
        return winner

X=dataset;batch_size=25
def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X

def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W


# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()


if __name__ == '__main__':

    # 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

    a = data.split(',')
    dataset = np.mat([[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)])

    dataset = dataset.reshape([20,3])
    dataset_old = dataset.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    som.train()
    res = som.train_result()
    classify = {}
    for i, win in enumerate(res):
        if not classify.get(win[0]):
            classify.setdefault(win[0], [i])
        else:
            classify[win[0]].append(i)
    C = []  # 未归一化的数据分类结果
    D = []  # 归一化的数据分类结果
    for i in classify.values():
        C.append(dataset_old[i].tolist())
        D.append(dataset[i].tolist())
    draw(C)
    draw(D)
    
    
import numpy as np
import pylab as pl


class SOM(object):
    def __init__(, X, output, iteration, batch_size):
        """
        :param X:  形状是N*D， 输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        X = X
        output = output
        iteration = iteration
        batch_size = batch_size
        W = np.random.rand(X.shape[1], output[0] * output[1])
        print(W.shape)

iteration=1
def GetN(t):
    """
    :param t:时间t, 这里用迭代次数来表示时间
    :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
    """
    a = min(output)
    return int(a - float(a) * t / iteration)

def Geteta(t, n):
    """
    :param t: 时间t, 这里用迭代次数来表示时间
    :param n: 拓扑距离
    :return: 返回学习率，
    """
    return np.power(np.e, -n) / (t + 2)

def updata_W(X, t, winner):
    N = GetN(t)
    for x, i in enumerate(winner):
        to_update = getneighbor(i[0], N)
        for j in range(N + 1):
            e = Geteta(t, j)
            for w in to_update[j]:
                W[:, w] = np.add(W[:, w], e * (X[x, :] - W[:, w]))

def getneighbor(index, N):
    """
    :param index:获胜神经元的下标
    :param N: 邻域半径
    :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
    """
    a, b = output
    length = a * b

    def distence(index1, index2):
        i1_a, i1_b = index1 // a, index1 % b
        i2_a, i2_b = index2 // a, index2 % b
        return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

    ans = [set() for i in range(N + 1)]
    for i in range(length):
        dist_a, dist_b = distence(i, index)
        if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
    return ans

def train():
    """
    train_Y:训练样本与形状为batch_size*(n*m)
    winner:一个一维向量，batch_size个获胜神经元的下标
    :return:返回值是调整后的W
    """
    count = 0
    while iteration > count:
        train_X = X[np.random.choice(X.shape[0], batch_size)]
        normal_W(W)
        normal_X(train_X)
        train_Y = train_X.dot(W)
        winner = np.argmax(train_Y, axis=1).tolist()
        updata_W(train_X, count, winner)
        count += 1
    return W
X.shape
batch_size =20
def train_result():
    normal_X(X)
    train_Y = X.dot(W)
    winner = np.argmax(train_Y, axis=1).tolist()
    print(winner)
    return winner


def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X


def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W


# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()


if __name__ == '__main__':

    # 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

    a = data.split(',')
    dataset = np.mat([[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)])
    dataset_old = dataset.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    som.train()
    res = som.train_result()
    classify = {}
    for i, win in enumerate(res):
        if not classify.get(win[0]):
            classify.setdefault(win[0], [i])
        else:
            classify[win[0]].append(i)
    C = []  # 未归一化的数据分类结果
    D = []  # 归一化的数据分类结果
    for i in classify.values():
        C.append(dataset_old[i].tolist())
        D.append(dataset[i].tolist())
    draw(C)
    draw(D)

import numpy as np
y1=np.random.randint(2,10,(5,3))
print ("排序列表：", y1)
np.random.shuffle(y1)
print ("随机排序列表：", y1)
