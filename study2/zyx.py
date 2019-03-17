n = 11


def f(n):
    L = list(range(n, 0, -1))
    output = []
    while (L):
        a = L.pop()
        if L:
            b = L.pop()
            L.insert(0, b)

        output += [a]
    return output


output = f(10)
for x in output:
    print(x, end=' ')
print()


def f(n):
    L = list(range(n, 0, -1))
    output = []
    while (L):
        a = L.pop()
        if L:
            b = L.pop()
            L.insert(0, b)
        output += [a]

    for x in output:
        print(x, end=' ')
    print()


f(10)

a = f(11)

bool(L)

L[-1::-1]

__author__ = 'wym'

import numpy as np
import random

n = random.randint(1, 100000)
a = random.randint(-100000, 100000)

x = np.random.randint(-100000, 100000, (1, n))[0]
x = np.append(a, x).tolist()

node_map = np.zeros((n, n))
for t1, x1 in enumerate(x):
    for t2, x2 in enumerate(x):
        node_map[t1, t2] = abs(x1 - x2)


# -*-coding:utf-8 -*-
class DijkstraExtendPath():
    def __init__(self, node_map):
        self.node_map = node_map
        self.node_length = len(node_map)
        self.used_node_list = []
        self.collected_node_dict = {}

    def __call__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self._init_dijkstra()
        return self._format_path()

    def _init_dijkstra(self):
        self.used_node_list.append(self.from_node)
        self.collected_node_dict[self.from_node] = [0, -1]
        for index1, node1 in enumerate(self.node_map[self.from_node]):
            if node1:
                self.collected_node_dict[index1] = [node1, self.from_node]
        self._foreach_dijkstra()

    def _foreach_dijkstra(self):
        if len(self.used_node_list) == self.node_length - 1:
            return
        for key, val in self.collected_node_dict.items():  # 遍历已有权值节点
            if key not in self.used_node_list and key != to_node:
                self.used_node_list.append(key)
            else:
                continue
            for index1, node1 in enumerate(self.node_map[key]):  # 对节点进行遍历
                # 如果节点在权值节点中并且权值大于新权值
                if node1 and index1 in self.collected_node_dict and self.collected_node_dict[index1][0] > node1 + val[
                    0]:
                    self.collected_node_dict[index1][0] = node1 + val[0]  # 更新权值
                    self.collected_node_dict[index1][1] = key
                elif node1 and index1 not in self.collected_node_dict:
                    self.collected_node_dict[index1] = [node1 + val[0], key]
        self._foreach_dijkstra()

    def _format_path(self):
        node_list = []
        temp_node = self.to_node
        node_list.append((temp_node, self.collected_node_dict[temp_node][0]))
        while self.collected_node_dict[temp_node][1] != -1:
            temp_node = self.collected_node_dict[temp_node][1]
            node_list.append((temp_node, self.collected_node_dict[temp_node][0]))
        node_list.reverse()
        return node_list


def set_node_map(node_map, node, node_list):
    for x, y, val in node_list:
        node_map[node.index(x)][node.index(y)] = node_map[node.index(y)][node.index(x)] = val


if __name__ == "__main__":
    node = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    node_list = [('A', 'F', 9), ('A', 'B', 10), ('A', 'G', 15), ('B', 'F', 2),
                 ('G', 'F', 3), ('G', 'E', 12), ('G', 'C', 10), ('C', 'E', 1),
                 ('E', 'D', 7)]
    node_map = [[0 for val in range(len(node))] for val in range(len(node))]
    set_node_map(node_map, node, node_list)
    # A -->; D
    from_node = node.index('A')
    to_node = node.index('D')
    dijkstrapath = DijkstraExtendPath(node_map)
    path = dijkstrapath(from_node, to_node)
    print
    path

import numpy as np
import random

n = random.randint(1, 100000)
a = random.randint(-100000, 100000)

x = np.random.randint(-100000, 100000, (1, n))[0]

node_map = np.zeros((n, n))
for t1, x1 in enumerate(x):
    for t2, x2 in enumerate(x):
        node_map[t1, t2] = abs(x1 - x2)

#################floyd
import numpy as np

N = 4
M = 100
edge = np.mat([[0, 2, 6, 4], [M, 0, 3, M], [7, M, 0, 1], [5, M, 12, 0]])
A = edge[:]
path = np.zeros((N, N))

for a in range(N):
    for b in range(N):
        for c in range(N):
            if (A[b, a] + A[a, c] < A[b, c]):
                A[b, c] = A[b, a] + A[a, c]
                path[b][c] = path[b][c] * 10 + a

######################### dijkstra
N = 4
M = 100
W = np.mat([[0, 2, 6, 4], [M, 0, 3, M], [7, M, 0, 1], [5, M, 12, 0]])
n = 4
st = 1;
e = 0
D = np.array(W[st, :])[0]
visit = np.ones(n)
visit[st] = 0
parent = np.zeros(n)  # 记录每个节点的上一个节点

path = []
for i in range(n - 1):
    temp = []
    for j in range(n):
        if visit[j]:
            temp += [D[j]]
        else:
            temp += [np.inf]
    print(temp)
    value = np.min(temp)
    index = np.argmin(temp)
    visit[index] = 0
    for k in range(n):
        if D[k] > D[index] + W[index, k]:
            D[k] = D[index] + W[index, k]
            parent[k] = parent[k] * 10 + index
    print(D)
distance = D[e]


class Graph():
    def __init__(self, nodeNum, sides, direction=False):
        self.nodeNum = nodeNum  # 顶点
        self.amatrix = [[0] * (nodeNum + 1) for i in range(nodeNum + 1)]  # 邻接矩阵
        for side in sides:
            u, v, w = side
            if (direction):
                self.amatrix[u][v] = w
            else:
                self.amatrix[u][v] = w
                self.amatrix[v][u] = w


def dfs(graph, v):
    visitnodes.append(v)
    for j in range(graph.nodeNum + 1):
        if ((graph.amatrix[v][j] > 0) and (j not in visitnodes)):
            dfs(graph, j)


def dfs(graph, v):
    visitnodes.append(v)
    for j in range(graph.nodeNum + 1):
        if ((graph.amatrix[v][j] > 0) and (j not in visitnodes)):
            dfs(graph, j)


nodeNum = 9
sides = [[1, 2, 1], [2, 3, 1], [3, 4, 1], [2, 5, 1], [4, 6, 1], [5, 7, 1], [5, 8, 1], [8, 9, 1]]
graph = Graph(9, [[1, 2, 1], [2, 3, 1], [3, 4, 1], [2, 5, 1], [4, 6, 1], [5, 7, 1], [5, 8, 1], [8, 9, 1]])
graph.amatrix

visitnodes = []
dfs(graph, 1)
print(visitnodes)

[1, 2, 3, 4, 6, 5, 7, 8, 9]

import queue


def bfs(adj, start):
    visited = set()
    q = queue.Queue()
    q.put(start)
    while not q.empty():
        u = q.get()
        print(u)
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)


graph = {1: [4, 2], 2: [3, 4], 3: [4], 4: [5]}
bfs(graph, 1)
start = 1
v = 2

import queue

q1 = queue.Queue(4)    # 4是队列长度
q2 = queue.LifoQueue()  # 栈
q3 = queue.PriorityQueue()  # heap
q1.put(2)
q1.get()
q1.qsize()
q1.empty()
q1.full()
list(q1.queue)

from heapq import heappush, heappop
import heapq
q4=[1,5,3,2,9,5]
heappush(q4,1)
print(q4)
heappop(q4)


def _siftdown(heap):
    startpos=0
    pos=len(heap)-1
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem
a=[1,4,5,23,1,4]
_siftdown(a)










# coding=utf8
G = [
    {5: 10, 1: 28},  # 0
    {0: 28, 6: 14, 2: 16},  # 1
    {1: 16, 3: 12},  # 2
    {2: 12, 4: 22, 6: 18},  # 3
    {3: 22, 5: 25, 6: 24},  # 4
    {0: 10, 4: 25},  # 5
    {1: 14, 3: 18, 4: 24}  # 6
]

##########################最小生成树
import heapq


def prim(G):
    n = len(G)  # 这个图的顶点个数，prim算法的边数就是顶点数减一
    v = 0  # 从第一个顶点开始
    s = {v}  # 这个集合避免重复的顶点被重复执行，造成无限循环
    edges = []  # 存放边的值
    # 存放结果
    res = []
    for _ in range(n - 1):
        # 对字典进行解包
        for u, w in G[v].items():
            print(u, w)
            heapq.heappush(edges, (w, v, u))
        while edges:
            w, p, q = heapq.heappop(edges)

            if q not in s:
                s.add(q)
                res.append(((p, q), w))
                v = q
                break
    return res


prim(G)

import numpy as np
import random

n = random.randint(1, 100000)
n = 100
a = random.randint(-100000, 100000)
x = np.random.randint(-100000, 100000, size=n)
x = np.append(a, x)

y = [(w, u) for u, w in enumerate(x)]
y.sort()
if abs(y[0][0] - a) > abs(y[-1][0] - a):
    y.pop(0)
else:
    y.pop()
if abs(y[0][0] - a) > abs(y[-1][0] - a):
    distance = abs(y[0][0] - y[-1][0]) + abs(y[-1][0])
else:
    distance = abs(y[0][0] - y[-1][0]) + abs(y[0][0])


# node_map = np.zeros((n,n))
# for t1,x1 in enumerate(x):
#     for t2, x2 in enumerate(x):
#         node_map[t1,t2]=abs(x1-x2)

def prim(G):
    n = len(x) - 1  # 这个图的顶点个数，prim算法的边数就是顶点数减一
    v = 0  # 从第一个顶点开始
    s = {v}  # 这个集合避免重复的顶点被重复执行，造成无限循环
    edges = []  # 存放边的值
    # 存放结果
    res = []
    for _ in range(n - 1):
        # 对字典进行解包
        for u, w in enumerate(x):
            # print(u,w)
            if u != v:
                w = abs(w - x[v])
                heapq.heappush(edges, (w, v, u))
        while edges:
            w, p, q = heapq.heappop(edges)

            if q not in s:
                s.add(q)
                res.append(((p, q), w))
                v = q
                break
    return res


import queue
import stack
import myfunc

myfunc.search('ShowProcess')
