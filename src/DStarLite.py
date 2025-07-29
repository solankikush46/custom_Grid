# DStarLite.py

import heapq
import math

class DStarLite:
    """
    An optimized D* Lite pathfinding algorithm that supports dynamic edge costs
    and multiple goal destinations.
    """
    def __init__(self, grid, start, goals, cost_function):
        if not goals:
            raise ValueError("Goals list cannot be empty.")

        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.cost_function = cost_function

        self.real_goals = goals
        self.s_start = start
        self.s_last = start
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        
        self.U = []
        self.U_members = set()

        for r in range(self.height):
            for c in range(self.width):
                self.g[(c, r)] = float('inf')
                self.rhs[(c, r)] = float('inf')
        
        for goal in self.real_goals:
            self.rhs[goal] = 0
            heapq.heappush(self.U, (self._calculate_key(goal), goal))
            self.U_members.add(goal)

    def _heuristic(self, s1, s2):
        """
        Calculates the Euclidean distance heuristic between two points.
        """
        dx = s1[0] - s2[0]
        dy = s1[1] - s2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_key(self, s):
        g_s = self.g.get(s, float('inf'))
        rhs_s = self.rhs.get(s, float('inf'))
        min_g_rhs = min(g_s, rhs_s)
        return (min_g_rhs + self._heuristic(self.s_start, s) + self.k_m, min_g_rhs)

    def _update_node(self, u):
        is_consistent = self.g.get(u, float('inf')) == self.rhs.get(u, float('inf'))
        is_in_queue = u in self.U_members

        if not is_consistent and u in self.U_members:
            # Re-prioritize in the heap
            self.U = [(self._calculate_key(v), v) for k, v in self.U if v == u] + \
                     [(k, v) for k, v in self.U if v != u]
            heapq.heapify(self.U)
        elif not is_consistent and not is_in_queue:
            heapq.heappush(self.U, (self._calculate_key(u), u))
            self.U_members.add(u)
        elif is_consistent and is_in_queue:
            self.U_members.remove(u)
            self.U = [(k, v) for k, v in self.U if v != u]
            heapq.heapify(self.U)

    def _compute_shortest_path(self):
        while (self.U and
               (heapq.nsmallest(1, self.U)[0][0] < self._calculate_key(self.s_start) or
            self.rhs.get(self.s_start, float('inf')) != self.g.get(self.s_start, float('inf')))):
            
            k_old, u = heapq.heappop(self.U)
            self.U_members.remove(u)

            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                for s_prime in self._get_neighbors(u):
                    self.rhs[s_prime] = min(self.rhs.get(s_prime, float('inf')),
                                          self._get_edge_cost(s_prime, u) + self.g.get(u, float('inf')))
                    self._update_node(s_prime)
            else:
                self.g[u] = float('inf')
                for s_prime in self._get_neighbors(u) + [u]:
                    if self.rhs.get(s_prime, float('inf')) == self._get_edge_cost(s_prime, u) + self.g.get(u, float('inf')):
                         if s_prime not in self.real_goals:
                            min_rhs = float('inf')
                            for s_hat in self._get_neighbors(s_prime):
                                min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
                            self.rhs[s_prime] = min_rhs
                    self._update_node(s_prime)

    def _get_neighbors(self, u):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1,-1), (-1,1), (1,-1), (1,1)]: # 8-way
            nx, ny = u[0] + dx, u[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def _get_edge_cost(self, u, v):
        # Cost includes move distance + destination cost
        move_cost = math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2) # Diagonal moves cost more
        return move_cost + self.cost_function(v)

    def move_and_replan(self, new_start):
        if new_start == self.s_start: return
        self.k_m += self._heuristic(self.s_last, new_start)
        self.s_last = new_start
        self.s_start = new_start
        self._compute_shortest_path()

    def update_costs(self, changed_nodes):
        for u in changed_nodes:
            # Re-calculate rhs for neighbors of changed nodes
            for s_prime in self._get_neighbors(u):
                min_rhs = float('inf')
                for s_hat in self._get_neighbors(s_prime):
                    min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
                self.rhs[s_prime] = min_rhs
                self._update_node(s_prime)
        self._compute_shortest_path()

    def get_path_to_goal(self):
        if self.g.get(self.s_start, float('inf')) == float('inf'): return []
        path, curr = [self.s_start], self.s_start
        
        while curr not in self.real_goals:
            min_cost, next_node = float('inf'), None
            for s_prime in self._get_neighbors(curr):
                cost = self._get_edge_cost(curr, s_prime) + self.g.get(s_prime, float('inf'))
                if cost < min_cost:
                    min_cost, next_node = cost, s_prime
            
            if next_node is None or next_node in path: return [] # No path or stuck in loop
            path.append(next_node)
            curr = next_node
        return path
