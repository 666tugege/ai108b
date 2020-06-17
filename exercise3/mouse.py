class Node:
    def __init__(self):
        self.x=0
        self.y=0
        self.next=None

def push(x,y):
    global top
    new_node=Node()
    new_node.x=x
    new_node.y=y
    new_node.next=top
    top=new_node
    
def pop():
    global top
    top=top.next
    
def check_point(x,y,ex,ey):
    if x==ex and y==ey:
        return 1
    else:
        return 0
    
def next_step(MAZE,ex,ey):
    a=top.x
    b=top.y
    ex=ex
    ey=ey
    if MAZE[a-1][b]==0:
        push(a-1,b)
        MAZE[a-1][b]=2
    elif MAZE[a][b-1]==0:
        push(a,b-1)
        MAZE[a][b-1]=2
    elif MAZE[a+1][b]==0:
        push(a+1,b)
        MAZE[a+1][b]=2
    elif MAZE[a][b+1]==0:
        push(a,b+1)
        MAZE[a][b+1]=2
    else:
        pop()
        
def find_road(MAZE,x,y,ex,ey):
    global top
    a=ex
    b=ey
    head=Node()
    head.x=x
    head.y=y
    head.next=None
    top=head
    while check_point(top.x,top.y,a,b)==0:
        next_step(MAZE,a,b)
        
    print('老鼠走过的路为：')
    for i in range(10):
        for j in range(12):
            print(MAZE[i][j],end='')
        print()
            
    
MAZE=[[1,1,1,1,1,1,1,1,1,1,1,1],\
      [1,0,0,0,1,1,1,1,1,1,1,1],\
      [1,1,1,0,1,1,0,0,0,0,1,1],\
      [1,1,1,0,1,1,0,1,1,0,1,1],\
      [1,1,1,0,0,0,0,1,1,0,1,1],\
      [1,1,1,0,1,1,0,1,1,0,1,1],\
      [1,1,1,0,1,1,0,1,1,0,1,1],\
      [1,1,1,1,1,1,0,1,1,0,1,1],\
      [1,1,0,0,0,0,0,0,1,0,0,1],\
      [1,1,1,1,1,1,1,1,1,1,1,1]]
x=1
y=1
ex=8
ey=10
print('='*20)
print('迷宫为:')
for i in range(10):
        for j in range(12):
            print(MAZE[i][j],end='')
        print()
print('='*20)
find_road(MAZE,x,y,ex,ey)
print('='*20)
while top!=None:
    MAZE[top.x][top.y]=3
    top=top.next
for i in range(10):
        for j in range(12):
            print(MAZE[i][j],end='')
        print()
while top!=None:
    MAZE[top.x][top.y]=3
    top=top.next
for i in range(10):
        for j in range(12):
            print(MAZE[i][j],end='')
        print()
