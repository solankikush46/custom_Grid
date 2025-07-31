import heapq
import math

INF = float('inf')
SQRT2 = math.sqrt(2)

class DStarLite:
    """
    Standalone D* Lite supporting 8-way movement (4 cardinals + 4 diagonals),
    with customizable edge-costs via an external cost_function.
    Now supports multiple goal positions.
    """

    def __init__(self, width, height, start, goal, cost_function, known_obstacles=None, heuristic=None):
        # Grid dimensions and start position
        self.width = width
        self.height = height
        self.start = start  # (x, y) tuple for the start position

        # Determine goal set (allow multiple goals)
        if isinstance(goal, tuple):
            # If 'goal' is a tuple of two numbers, treat as single goal; 
            # if it's a tuple of tuples (e.g., multiple goals), treat accordingly.
            if len(goal) == 2 and all(isinstance(coord, (int, float)) for coord in goal):
                self.goals = {goal}
            else:
                # Tuple containing multiple goal coordinates
                self.goals = set(goal)
        elif isinstance(goal, list) or isinstance(goal, set):
            # List or set of goal coordinates
            self.goals = set(goal)
        else:
            # Any other format (single goal)
            self.goals = {goal}

        # Backward compatibility: if only one goal, also store it in self.goal
        self.goal = None
        if len(self.goals) == 1:
            self.goal = next(iter(self.goals))

        # External cost function: maps a cell (x,y) -> float cost (e.g., battery cost)
        self.cost_function = cost_function
        # Static obstacles: set of impassable cells
        self.obstacles = set(known_obstacles) if known_obstacles else set()

        # Heuristic for distance estimation (default: Chebyshev distance for grid)
        if heuristic:
            self.h = heuristic
        else:
            self.h = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]))

        # Initialize g and rhs for all cells to infinity
        self.g   = { (x, y): INF for x in range(width) for y in range(height) }
        self.rhs = { (x, y): INF for x in range(width) for y in range(height) }

        # Priority queue (U) and lookup for entries, plus key modifier (km)
        self.U = []          # heap of (key, state)
        self.U_entry = {}    # maps state -> current key in queue (for lazy removal)
        self.km = 0          # key modifier (increased when start moves)

        # Initialize all goal states as sources with rhs = 0
        for g in self.goals:
            self.rhs[g] = 0
            self._push(g, self._calc_key(g))

    def _calc_key(self, s):
        # Calculate the priority key for state s
        g_rhs = min(self.g[s], self.rhs[s])
        # Key = (min(g, rhs) + heuristic_distance(start, s) + km, min(g, rhs))
        return (g_rhs + self.h(self.start, s) + self.km, g_rhs)

    def _push(self, s, key):
        # Push state s into the priority queue with the given key
        heapq.heappush(self.U, (key, s))
        self.U_entry[s] = key

    def _pop(self):
        # Pop the state with smallest key off the queue (skipping stale entries)
        while self.U:
            key, s = heapq.heappop(self.U)
            if self.U_entry.get(s) == key:  # ensure this is the latest entry for s
                del self.U_entry[s]
                return s
        return None

    def _top_key(self):
        # Peek at the smallest key in the queue (after removing any stale entries)
        while self.U:
            key, s = self.U[0]
            if self.U_entry.get(s) != key:
                # Remove outdated entry
                heapq.heappop(self.U)
                continue
            return key
        return (INF, INF)

    def in_bounds(self, s):
        # Check if coordinate s = (x,y) lies within grid bounds
        x, y = s
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, s):
        # Generate all valid neighboring coordinates (8-directional moves allowed)
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        result = []
        for dx, dy in directions:
            nb = (s[0] + dx, s[1] + dy)
            if self.in_bounds(nb) and nb not in self.obstacles:
                result.append(nb)
        return result

    def cost(self, u, v):
        # Compute travel cost from u to v, including terrain and external costs
        if not (self.in_bounds(u) and self.in_bounds(v)):
            return INF  # outside grid bounds
        if u in self.obstacles or v in self.obstacles:
            return INF  # hitting an obstacle is not allowed
        # Base movement cost (diagonal vs straight)
        base_cost = SQRT2 if (u[0] != v[0] and u[1] != v[1]) else 1.0
        # Add external cost for entering v (from cost_function)
        return base_cost + self.cost_function(v)

    def succ(self, s):
        # Successors of s (identical to neighbors in an undirected grid)
        return self.neighbors(s)

    def pred(self, s):
        # Predecessors of s (also identical to neighbors here)
        return self.neighbors(s)

    def update_vertex(self, u):
        # Update the rhs value of state u and adjust its queue entry
        if u not in self.goals:
            # If u is not a goal, compute the minimum cost to go to any neighbor plus that neighbor's g value
            nbrs = self.succ(u)
            self.rhs[u] = min((self.cost(u, s_prime) + self.g[s_prime]) for s_prime in nbrs) if nbrs else INF
        # Remove any existing entry for u in the queue (we will re-add if needed)
        if u in self.U_entry:
            del self.U_entry[u]
        # If u is inconsistent (g != rhs), push it with updated key
        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def compute_shortest_path(self):
        # Compute or repair the shortest path tree until the start is consistent or no better paths remain
        while (self._top_key() < self._calc_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            u = self._pop()
            if u is None:
                break  # queue is empty, no path found (start might be unreachable)
            if self.g[u] > self.rhs[u]:
                # Improved estimate for u (or u is a goal with rhs = 0)
                self.g[u] = self.rhs[u]
                # Propagate improvement to all predecessors of u
                for p in self.pred(u):
                    self.update_vertex(p)
            else:
                # No longer the best path for u, revert g[u] and update neighbors
                self.g[u] = INF
                # u itself and all predecessors may need to recompute their rhs
                for p in self.pred(u) + [u]:
                    self.update_vertex(p)

    def set_obstacle(self, cell):
        # Introduce a new obstacle at given cell and update affected states
        if cell not in self.obstacles:
            self.obstacles.add(cell)
            # When a new obstacle is added, update that cell and its neighbors
            for p in self.pred(cell) + [cell]:
                self.update_vertex(p)

    def get_shortest_path(self):
        # Reconstruct a path from the start to the nearest goal using computed g-values
        if self.g[self.start] == INF:
            return []  # no path to any goal
        path = [self.start]
        s = self.start
        # Follow the tree (greedily by cost) until a goal is reached
        while s not in self.goals:
            nbrs = self.succ(s)
            if not nbrs:
                return []  # dead end
            # Choose the neighbor with lowest (cost + g[]) sum 
            s = min(nbrs, key=lambda s_prime: self.cost(s, s_prime) + self.g[s_prime])
            if s in path:
                print("[ERROR] Cycle detected")
                return []  # detected a cycle (should not happen if costs are positive)
            path.append(s)
        return path
