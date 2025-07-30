# weighted_astar.py

import heapq
from typing import Tuple, List, Optional, Dict, Any

def weighted_astar(
    start: Tuple[int,int],
    goals: List[Tuple[int,int]],
    grid: List[List[Any]],
    cost_fn,
    w: float = 2.0
) -> Optional[List[Tuple[int,int]]]:
    """
    Weighted A* on a uniform-cost grid with arbitrary cost_fn.
    f(n) = g(n) + w * h(n)
    Returns the full cell-by-cell path or None if unreachable.
    """
    rows, cols = len(grid), len(grid[0])

    def h(a: Tuple[int,int]) -> float:
        # Octile distance to the closest goal
        best = float('inf')
        for gx, gy in goals:
            dx, dy = abs(a[0]-gx), abs(a[1]-gy)
            # octile
            d = (dx + dy) + (2**0.5 - 2)*min(dx, dy)
            best = min(best, d)
        return best

    open_heap: List[Tuple[float, Tuple[int,int]]] = []
    g_score: Dict[Tuple[int,int], float] = {start: 0}
    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}

    heapq.heappush(open_heap, (w*h(start), start))

    while open_heap:
        f_current, current = heapq.heappop(open_heap)
        if current in goals:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < cols and 0 <= ny < rows): 
                continue
            if not grid[ny][nx]:
                continue
            tentative_g = g_score[current] + cost_fn((nx, ny))
            if tentative_g < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = current
                f = tentative_g + w*h((nx, ny))
                heapq.heappush(open_heap, (f, (nx, ny)))

    return None
