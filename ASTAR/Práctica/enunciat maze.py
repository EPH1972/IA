from pyamaze import maze, agent, COLOR, textLabel
from queue import PriorityQueue

def h(celda1, celda2):
    """Heurística: Distancia Manhattan"""
    x1, y1 = celda1
    x2, y2 = celda2
    return abs(x1 - x2) + abs(y1 - y2)

def aStar(m):
    start = (m.rows, m.cols)
    v_priority = PriorityQueue()
    # (f_score, h_score, celda)
    v_priority.put((h(start, (1, 1)), h(start, (1, 1)), start))
    
    camino_recorrido = {}
    g_score = {celda: float('inf') for celda in m.grid}
    g_score[start] = 0
    f_score = {celda: float('inf') for celda in m.grid}
    f_score[start] = h(start, (1, 1))

    while not v_priority.empty():
        currCell = v_priority.get()[2]
        if currCell == (1, 1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d] == 1:
                if d == 'E': nextCell = (currCell[0], currCell[1] + 1)
                if d == 'W': nextCell = (currCell[0], currCell[1] - 1)
                if d == 'N': nextCell = (currCell[0] - 1, currCell[1])
                if d == 'S': nextCell = (currCell[0] + 1, currCell[1])

                temp_g_score = g_score[currCell] + 1
                temp_f_score = temp_g_score + h(nextCell, (1, 1))

                if temp_f_score < f_score[nextCell]:
                    g_score[nextCell] = temp_g_score
                    f_score[nextCell] = temp_f_score
                    v_priority.put((temp_f_score, h(nextCell, (1, 1)), nextCell))
                    camino_recorrido[nextCell] = currCell
    
    # Reconstruir camino final
    fwdPath = {}
    cell = (1, 1)
    while cell != start:
        fwdPath[camino_recorrido[cell]] = cell
        cell = camino_recorrido[cell]
    return camino_recorrido, fwdPath

# --- Ejecución ---
m = maze(15, 15)
m.CreateMaze(1, 1, loopPercent=100) # Genera laberinto
camino_recorrido, path = aStar(m)

# Agente amarillo: muestra el orden de exploración de A*
a_busqueda = agent(m, footprints=True, color=COLOR.yellow, filled=True, shape='square')
# Agente cyan: muestra el camino óptimo final
a_camino = agent(m, footprints=True, color=COLOR.cyan, filled=True)

m.tracePath({a_busqueda: camino_recorrido}, delay=20)
m.tracePath({a_camino: path}, delay=50)

textLabel(m, 'Celdas Exploradas', len(camino_recorrido))
textLabel(m, 'Pasos Totales', len(path) + 1)

m.run()