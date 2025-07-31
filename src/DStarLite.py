# DStarLite.py

import heapq
import math

INF = float('inf')
SQRT2 = math.sqrt(2)

class DStarLite:
    """
    Standalone D* Lite supporting 8-way movement (4 cardinals + 4 diagonals),
    with customizable edge-costs via an external cost_function.
    """

    def __init__(self, width, height, start, goal, cost_function, known_obstacles=None, heuristic=None):
        # Grid dimensions and start/goal positions
        self.width  = width
        self.height = height
        self.start  = start    # (x,y)
        self.goal   = goal     # (x,y)

        # External cost function: cost_function(u,v) -> float
        self.cost_function = cost_function

        # obstacle set: cells with infinite cost (static impassables)
        self.obstacles = set(known_obstacles) if known_obstacles else set()

        # Heuristic: default = Chebyshev distance (admissible for 8-way costs)
        if heuristic:
            self.h = heuristic
        else:
            self.h = lambda a, b: max(abs(a[0]-b[0]), abs(a[1]-b[1]))

        # g & rhs arrays
        self.g   = { (x,y): INF for x in range(width) for y in range(height) }
        self.rhs = { (x,y): INF for x in range(width) for y in range(height) }

        # priority queue & key modifier
        self.U       = []   # heap of (key, state)
        self.U_entry = {}   # maps state -> key for lazy removal
        self.km      = 0

        # initialize goal
        self.rhs[self.goal] = 0
        self._push(self.goal, self._calc_key(self.goal))

    def _calc_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return ( g_rhs + self.h(self.start, s) + self.km, g_rhs )

    def _push(self, s, key):
        heapq.heappush(self.U, (key, s))
        self.U_entry[s] = key

    def _pop(self):
        while self.U:
            key, s = heapq.heappop(self.U)
            if self.U_entry.get(s) == key:
                del self.U_entry[s]
                return s
        return None

    def _top_key(self):
        while self.U:
            key, s = self.U[0]
            if self.U_entry.get(s) != key:
                heapq.heappop(self.U)
                continue
            return key
        return (INF, INF)

    def in_bounds(self, s):
        x,y = s
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, s):
        # 8 directions
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        result = []
        for dx,dy in dirs:
            nb = (s[0]+dx, s[1]+dy)
            if self.in_bounds(nb) and nb not in self.obstacles:
                result.append(nb)
        return result

    def cost(self, u, v):
        # ∞ if OOB or static obstacle
        if not (self.in_bounds(u) and self.in_bounds(v)):
            return INF
        if u in self.obstacles or v in self.obstacles:
            return INF
        # move cost: 1 or sqrt2
        base = SQRT2 if (u[0]!=v[0] and u[1]!=v[1]) else 1.0
        # external cost at destination v
        return base + self.cost_function(v)

    def succ(self, s):
        return self.neighbors(s)

    def pred(self, s):
        return self.neighbors(s)

    def update_vertex(self, u):
        if u != self.goal:
            nbrs = self.succ(u)
            self.rhs[u] = min((self.cost(u, sp) + self.g[sp]) for sp in nbrs) if nbrs else INF
        if u in self.U_entry:
            del self.U_entry[u]
        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def compute_shortest_path(self):
        while (self._top_key() < self._calc_key(self.start) or
               self.rhs[self.start] != self.g[self.start]):
            u = self._pop()
            if u is None:
                break
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for p in self.pred(u):
                    self.update_vertex(p)
            else:
                self.g[u] = INF
                for p in self.pred(u) + [u]:
                    self.update_vertex(p)

    def set_obstacle(self, cell):
        if cell not in self.obstacles:
            self.obstacles.add(cell)
            for p in self.pred(cell) + [cell]:
                self.update_vertex(p)

    def get_shortest_path(self):
        if self.g[self.start] == INF:
            return []
        path = [self.start]
        s = self.start
        while s != self.goal:
            nbrs = self.succ(s)
            if not nbrs:
                return []
            s = min(nbrs, key=lambda sp: self.cost(s, sp) + self.g[sp])
            if s in path:
                return []
            path.append(s)
        return path
