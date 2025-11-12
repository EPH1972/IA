import pyamaze as maze
import time
from queue import PriorityQueue

ROWS = 10
COLS = 10



trace = PriorityQueue()
count = 0
agentPos = f'{ROWS},{COLS}'

# def distance(cell1, cell2):

'''
def calcPos(dir):
    calcPos = agentPos.split(',')
    if dir == 'E':
        calcPos[0]+1
    if dir == 'W':
        calcPos[0]-1
    if dir == 'N':
        calcPos[1]-1
    if dir == 'S':
        calcPos[1]+1
    agentPos = f'calcPos[0],calcPos[1]'
    trace.put(f'({agentPos})')    
'''
'''
def nextPos(map):
    while True:
        dir = map.get(agentPos)
        newPos = calcPos(dir)
        if dir not 0:
            agentPos = newPos
            break
    return agentPos
'''

def aStar(m):
    tracePath = ''
    map = m.maze_map
    print(map)
    while True:
        # agentPos = nextPos(map)
        if agentPos == '1,1':
            break
    return tracePath

m=maze.maze(ROWS,COLS)
m.CreateMaze()
pre_Astar = time.time()
path = aStar(m)
post_Astar = time.time()
print(post_Astar - pre_Astar)
a=maze.agent(m,footprints=True)
m.tracePath({a:path},delay=5)
m.run()