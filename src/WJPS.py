# WJPS.py

import heapq
from typing import Tuple, List, Optional, Dict, Any

# 8 directions: (dx, dy)
DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),  # cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
]

class JumpResult:
    def __init__(self, node: Tuple[int, int], dist: int):
        self.node = node
        self.dist = dist

def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height

def is_obstacle(x: int, y: int, grid: List[List[Any]]) -> bool:
    return not grid[y][x]

def has_forced_neighbor(x: int, y: int, dx: int, dy: int, grid: List[List[Any]], width: int, height: int) -> bool:
    # Checks for JPS forced neighbor conditions
    if dx == 0 or dy == 0:
        if dx != 0:
            if in_bounds(x, y+1, width, height) and is_obstacle(x-dx, y+1, grid) and not is_obstacle(x, y+1, grid):
                return True
            if in_bounds(x, y-1, width, height) and is_obstacle(x-dx, y-1, grid) and not is_obstacle(x, y-1, grid):
                return True
        else:
            if in_bounds(x+1, y, width, height) and is_obstacle(x+1, y-dy, grid) and not is_obstacle(x+1, y, grid):
                return True
            if in_bounds(x-1, y, width, height) and is_obstacle(x-1, y-dy, grid) and not is_obstacle(x-1, y, grid):
                return True
    else:
        if in_bounds(x-dx, y+dy, width, height) and is_obstacle(x-dx, y, grid) and not is_obstacle(x-dx, y+dy, grid):
            return True
        if in_bounds(x+dx, y-dy, width, height) and is_obstacle(x, y-dy, grid) and not is_obstacle(x+dx, y-dy, grid):
            return True
    return False

def jump(x: int, y: int, dx: int, dy: int, grid: List[List[Any]], width: int, height: int, goals: List[Tuple[int,int]], cost_fn) -> Optional[JumpResult]:
    start_cost = cost_fn((x,y))
    dist = 0
    while True:
        x += dx; y += dy; dist += 1
        if not in_bounds(x,y,width,height) or is_obstacle(x,y,grid):
            return None
        if (x,y) in goals:
            return JumpResult((x,y),dist)
        curr_cost = cost_fn((x,y))
        if curr_cost < start_cost:
            return JumpResult((x,y),dist)
        if has_forced_neighbor(x,y,dx,dy,grid,width,height):
            return JumpResult((x,y),dist)
        if dx!=0 and dy!=0 and (is_obstacle(x-dx,y,grid) or is_obstacle(x,y-dy,grid)):
            return None

def reconstruct_path(came_from: Dict[Tuple[int,int],Tuple[int,int]], current: Tuple[int,int]) -> List[Tuple[int,int]]:
    # Collect jump points
    points = [current]
    while current in came_from:
        current = came_from[current]
        points.append(current)
    points.reverse()
    # Expand full path into individual steps
    full: List[Tuple[int,int]] = []
    for (x0,y0),(x1,y1) in zip(points, points[1:]):
        dx = 0 if x1==x0 else (1 if x1>x0 else -1)
        dy = 0 if y1==y0 else (1 if y1>y0 else -1)
        x,y = x0,y0
        full.append((x,y))
        while (x,y)!=(x1,y1): x+=dx; y+=dy; full.append((x,y))
    full.append(points[-1])
    return full

def wjps(start: Tuple[int,int], goals: List[Tuple[int,int]], grid: List[List[Any]], cost_fn) -> Optional[List[Tuple[int,int]]]:
    height, width = len(grid), len(grid[0])
    def h(a: Tuple[int,int]) -> float:
        best = float('inf')
        for gx,gy in goals:
            dx = abs(a[0]-gx); dy = abs(a[1]-gy)
            d = (dx+dy) + (2**0.5-2)*min(dx,dy)
            best = min(best,d)
        return best
    openh: List[Tuple[float,Tuple[int,int]]] = []
    g_score: Dict[Tuple[int,int],float] = {start:0}
    came_from: Dict[Tuple[int,int],Tuple[int,int]] = {}
    heapq.heappush(openh,(h(start),start))
    while openh:
        _, cur = heapq.heappop(openh)
        if cur in goals:
            return reconstruct_path(came_from,cur)
        cx,cy = cur
        for dx,dy in DIRECTIONS:
            jr = jump(cx,cy,dx,dy,grid,width,height,goals,cost_fn)
            if not jr: continue
            succ = jr.node
            tentative = g_score[cur] + jr.dist * cost_fn(cur)
            if tentative < g_score.get(succ,float('inf')):
                g_score[succ] = tentative
                came_from[succ] = cur
                heapq.heappush(openh,(tentative+h(succ),succ))
    return None

def render_wjps(grid: List[List[Any]], start: Tuple[int,int], goals: List[Tuple[int,int]], cost_fn, cell_size: int=20) -> None:
    """
    Visualize grid, start/goals, and the WJPS path using Pygame.
    """
    import pygame
    pygame.init()
    rows,cols = len(grid),len(grid[0])
    screen = pygame.display.set_mode((cols*cell_size,rows*cell_size))
    clock = pygame.time.Clock()
    path = wjps(start,goals,grid,cost_fn)
    colors = {'free':(220,220,220),'obstacle':(50,50,50),'start':(0,255,0),'goal':(255,0,0),'path':(0,0,255)}
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
        screen.fill((0,0,0))
        for y in range(rows):
            for x in range(cols):
                c = colors['free'] if grid[y][x] else colors['obstacle']
                pygame.draw.rect(screen,c,(x*cell_size,y*cell_size,cell_size,cell_size))
        for gx,gy in goals:
            pygame.draw.rect(screen,colors['goal'],(gx*cell_size,gy*cell_size,cell_size,cell_size))
        sx,sy = start
        pygame.draw.rect(screen,colors['start'],(sx*cell_size,sy*cell_size,cell_size,cell_size))
        if path:
            for px,py in path:
                pygame.draw.rect(screen,colors['path'],(px*cell_size+cell_size//4,py*cell_size+cell_size//4,cell_size//2,cell_size//2))
        pygame.display.flip(); clock.tick(30)
    pygame.quit()
