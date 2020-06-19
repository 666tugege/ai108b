# 范例
```python
#广度优先深度BFS
#利用队列，先进先出，不唯一
graph={
       'A':['B','C'],
       'B':['A','C','D'],
       'C':['A','B','E','D'],
       'D':['B','C','E','F'],
       'E':['C','D'],
       'F':['D']
       
       }
def BFS(graph,s):
    queue=[]
    a=[]
    queue.append(s)
    seen=set()
    seen.add(s)
    while (len(queue)>0):
        vertex=queue.pop(0)
        nodes=graph[vertex]
        
        for node in nodes:
            if node not in seen:
                queue.append(node)
                seen.add(node)#标记已经访问过的元素
        a.append(vertex)
        print (vertex)
    return a
a=BFS(graph,'A')
```
找父节点
```python
def BFS(graph,s):
    queue=[]
    partent={}
    partent={s:None}
    queue.append(s)
    seen=set()
    seen.add(s)
    while (len(queue)>0):
        vertex=queue.pop(0)
        nodes=graph[vertex]
        
        for node in nodes:
            if node not in seen:
                queue.append(node)
                seen.add(node)#标记已经访问过的元素
                partent[node]=vertex
        
        print (vertex)
    return partent
partent=BFS(graph,'E')
v='B'
while (v!=None):
    print(v)
    v=partent[v]
```
深度搜索DFS，利用栈，后进先出
```python
def DFS(graph,s):
    stack=[]
    a=[]
    stack.append(s)
    seen=set()
    seen.add(s)
    while (len(stack)>0):
        vertex=stack.pop()
        nodes=graph[vertex]
        
        for node in nodes:
            if node not in seen:
                stack.append(node)
                seen.add(node)#标记已经访问过的元素
        a.append(vertex)
        print (vertex)
    return a
a=DFS(graph,'A')
```
带权重的最短路径
使用优先队列，每次找最小距离节点，更新所有节点的距离
```python
graph={
       'A':{'B':5,'C':1},
       'B':{'A':1,'C':2,'D':1},
       'C':{'A':1,'B':2,'E':8,'D':4},
       'D':{'B':1,'C':2,'E':3,'F':6},
       'E':{'C':8,'D':3},
       'F':{'D':6}
       
       }
graph['A']['B']
import heapq
import math

def init_distance(graph,s):
    distance={s:0}
    for vertex in graph:
        if vertex!=s:
            distance[vertex]=math.inf
    return distance
def dijkstra(graph,s):
    pqueue=[]
    heapq.heappush(pqueue,(0,s))#优先队列
    partent={}
    partent={s:None}
    
    seen=set()
    
    distance=init_distance(graph,s)
    while (len(pqueue)>0):
        pair=heapq.heappop(pqueue)
        vertex=pair[1]
        dist=pair[0]
        seen.add(vertex)
        nodes=graph[vertex].keys()
        
        for node in nodes:
            if node not in seen:
                if graph[vertex][node]+dist<distance[node]:
                    heapq.heappush(pqueue,(graph[vertex][node]+dist,node))
                    partent[node]=vertex
                    distance[node]=graph[vertex][node]+dist
        
        #print (vertex)
    return partent,distance
partent,distance=dijkstra(graph,'A')
w='D'
while(w!=None):
    print(w)
    w=partent[w]
```
