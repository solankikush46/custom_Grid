# path_planning.py

import heapq
import math
import pygame
import time
import random

# --- ALGORITHM AND ENVIRONMENT CLASSES (UNCHANGED FROM OPTIMIZED VERSION) ---

class DStarLite:
    def __init__(self, grid, start, goal, cost_function):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.cost_function = cost_function
        self.s_start = start
        self.s_goal = goal
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
        self.rhs[self.s_goal] = 0
        heapq.heappush(self.U, (self._calculate_key(self.s_goal), self.s_goal))
        self.U_members.add(self.s_goal)

    def _heuristic(self, s1, s2):
        return math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)

    def _calculate_key(self, s):
        return (min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))) +
                self._heuristic(self.s_start, s) + self.k_m,
                min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))))

    def _update_node(self, u):
        is_consistent = self.g.get(u, float('inf')) == self.rhs.get(u, float('inf'))
        is_in_queue = u in self.U_members
        if not is_consistent and not is_in_queue:
            heapq.heappush(self.U, (self._calculate_key(u), u))
            self.U_members.add(u)
        elif is_consistent and is_in_queue:
            self.U_members.remove(u)
            self.U = [(k, v) for k, v in self.U if v != u]
            heapq.heapify(self.U)

    def _pop_from_queue(self):
        if not self.U: return None, None
        key, node = heapq.heappop(self.U); self.U_members.remove(node); return key, node

    def _compute_shortest_path(self):
        while (self.U and (heapq.nsmallest(1, self.U)[0][0] < self._calculate_key(self.s_start) or
               self.rhs.get(self.s_start, float('inf')) != self.g.get(self.s_start, float('inf')))):
            k_old, u = self._pop_from_queue()
            if u is None: break
            k_new = self._calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u)); self.U_members.add(u); continue
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s_prime in self._get_neighbors(u):
                    self.rhs[s_prime] = min(self.rhs.get(s_prime, float('inf')), self._get_edge_cost(s_prime, u) + self.g.get(u, float('inf')))
                    self._update_node(s_prime)
            else:
                g_old = self.g.get(u, float('inf')); self.g[u] = float('inf')
                for s_prime in self._get_neighbors(u) + [u]:
                    if self.rhs.get(s_prime, float('inf')) == self._get_edge_cost(s_prime, u) + g_old:
                        if s_prime != self.s_goal:
                            min_rhs = float('inf')
                            for s_hat in self._get_neighbors(s_prime): min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
                            self.rhs[s_prime] = min_rhs
                    self._update_node(s_prime)

    def _get_neighbors(self, u):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = u[0] + dx, u[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def _get_edge_cost(self, u, v):
        return 1 + self.cost_function(v)

    def move_and_replan(self, new_start):
        if new_start == self.s_start: return
        self.k_m += self._heuristic(self.s_last, new_start); self.s_last = new_start; self.s_start = new_start
        self._compute_shortest_path()

    def update_costs(self, changed_edges):
        for u in changed_edges:
            for s_prime in self._get_neighbors(u) + [u]:
                if s_prime != self.s_goal:
                    min_rhs = float('inf')
                    for s_hat in self._get_neighbors(s_prime): min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
                    self.rhs[s_prime] = min_rhs
                self._update_node(s_prime)
        self._compute_shortest_path()

    def get_path_to_goal(self):
        path = []; curr = self.s_start
        for _ in range(self.width * self.height):
            if curr == self.s_goal: path.append(self.s_goal); return path
            path.append(curr)
            if self.g.get(curr, float('inf')) == float('inf'): return []
            min_cost = float('inf'); next_node = None
            for s_prime in self._get_neighbors(curr):
                cost = self._get_edge_cost(curr, s_prime) + self.g.get(s_prime, float('inf'))
                if cost < min_cost: min_cost = cost; next_node = s_prime
            if next_node is None: return []
            curr = next_node
        return []

class DynamicGridWorld:
    def __init__(self, width, height, loops_to_add=0):
        self.width = width; self.height = height; self.dynamic_costs = {}; self.impassable_nodes = set(); self.grid = self._generate_maze(); self._add_loops(loops_to_add)
    def _generate_maze(self):
        grid = [[1] * self.width for _ in range(self.height)]; stack = []; start_x, start_y = (1, 1); grid[start_y][start_x] = 0; stack.append((start_x, start_y))
        while stack:
            cx, cy = stack[-1]; neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.width-1 and 0 < ny < self.height-1 and grid[ny][nx] == 1: neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors); grid[ny][nx] = 0; grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0; stack.append((nx, ny))
            else: stack.pop()
        return grid
    def _add_loops(self, num_loops):
        potential_walls = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y][x] == 1:
                    if self.grid[y][x-1] == 0 and self.grid[y][x+1] == 0: potential_walls.append((x,y))
                    elif self.grid[y-1][x] == 0 and self.grid[y+1][x] == 0: potential_walls.append((x,y))
        random.shuffle(potential_walls)
        for i in range(min(num_loops, len(potential_walls))): wall_x, wall_y = potential_walls[i]; self.grid[wall_y][wall_x] = 0
    def get_cost(self, pos):
        if pos in self.impassable_nodes: return float('inf')
        return self.dynamic_costs.get(pos, 0)
    def draw(self, screen, camera_x, camera_y, zoom, colors):
        scr_width, scr_height = screen.get_size()
        zoomed_cell_size = int(8 * zoom) # Base cell size of 8
        if zoomed_cell_size < 1: zoomed_cell_size = 1 # Prevent cell size from being 0

        start_col = int(camera_x / zoomed_cell_size)
        end_col = start_col + int(scr_width / zoomed_cell_size) + 2
        start_row = int(camera_y / zoomed_cell_size)
        end_row = start_row + int(scr_height / zoomed_cell_size) + 2

        for y in range(max(0, start_row), min(self.height, end_row)):
            for x in range(max(0, start_col), min(self.width, end_col)):
                screen_x = x * zoomed_cell_size - camera_x
                screen_y = y * zoomed_cell_size - camera_y
                rect = pygame.Rect(screen_x, screen_y, zoomed_cell_size, zoomed_cell_size)
                if (x,y) in self.impassable_nodes: color = colors['red']
                elif self.grid[y][x] == 1: color = colors['black']
                elif (x, y) in self.dynamic_costs: color = colors['cost']
                else: color = colors['white']
                pygame.draw.rect(screen, color, rect)

# --- VISUALIZATION FUNCTION ---
def run_visual_test(width=51, height=41, loops=50, speed=15):
    pygame.init()
    screen_width, screen_height = 1280, 800 # Fixed screen size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("D* Lite Camera View: Use ARROWS to Pan, +/- to Zoom")
    clock = pygame.time.Clock()

    colors = {'white':(255,255,255),'black':(0,0,0),'blue':(0,0,255),'green':(0,255,0),'red':(255,0,0),'path':(30,144,255),'cost':(255,255,0)}
    font = pygame.font.Font(None, 30)

    # --- Loading Screen ---
    loading_font = pygame.font.Font(None, 50); text = loading_font.render("Generating Maze & Computing Path...", True, colors['red'])
    text_rect = text.get_rect(center=(screen_width/2, screen_height/2)); screen.fill(colors['black']); screen.blit(text, text_rect); pygame.display.flip()

    world = DynamicGridWorld(width, height, loops_to_add=loops)
    start_pos = (1, 1); goal_pos = (width - 2, height - 2); agent_pos = start_pos
    dstar = DStarLite(world.grid, agent_pos, goal_pos, world.get_cost)
    dstar._compute_shortest_path(); path = dstar.get_path_to_goal()

    # --- Camera and Zoom Setup ---
    zoom = 1.0
    base_cell_size = 8
    camera_x, camera_y = 0, 0
    pan_speed = 20

    running = True
    while running:
        zoomed_cell_size = int(base_cell_size * zoom)
        if zoomed_cell_size < 1: zoomed_cell_size = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                world_x = int((mx + camera_x) / zoomed_cell_size)
                world_y = int((my + camera_y) / zoomed_cell_size)
                pos = (world_x, world_y)

                if 0 <= world_x < width and 0 <= world_y < height and world.grid[world_y][world_x] == 0:
                    if event.button == 3:
                        if pos in world.impassable_nodes: world.impassable_nodes.remove(pos)
                        else: world.impassable_nodes.add(pos)
                    elif event.button == 1:
                        if pos in world.dynamic_costs: del world.dynamic_costs[pos]
                        else: world.dynamic_costs[pos] = 150.0
                    dstar.update_costs([pos]); path = dstar.get_path_to_goal()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: camera_x -= pan_speed
        if keys[pygame.K_RIGHT]: camera_x += pan_speed
        if keys[pygame.K_UP]: camera_y -= pan_speed
        if keys[pygame.K_DOWN]: camera_y += pan_speed
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]: zoom *= 1.1
        if keys[pygame.K_MINUS]: zoom /= 1.1
            
        if agent_pos != goal_pos and path and len(path) > 1:
            next_pos = path[1]; dstar.move_and_replan(next_pos); agent_pos = next_pos; path = dstar.get_path_to_goal()

        # Center camera on agent
        camera_x = agent_pos[0] * zoomed_cell_size - screen_width / 2
        camera_y = agent_pos[1] * zoomed_cell_size - screen_height / 2

        # --- Drawing ---
        screen.fill(colors['black'])
        world.draw(screen, camera_x, camera_y, zoom, colors)

        if path:
            path_points = []
            for p in path:
                screen_x = p[0] * zoomed_cell_size - camera_x + zoomed_cell_size / 2
                screen_y = p[1] * zoomed_cell_size - camera_y + zoomed_cell_size / 2
                path_points.append((screen_x, screen_y))
            if len(path_points) > 1: pygame.draw.lines(screen, colors['path'], False, path_points, max(1, int(2 * zoom)))

        agent_sx = agent_pos[0] * zoomed_cell_size - camera_x + zoomed_cell_size/2
        agent_sy = agent_pos[1] * zoomed_cell_size - camera_y + zoomed_cell_size/2
        goal_sx = goal_pos[0] * zoomed_cell_size - camera_x + zoomed_cell_size/2
        goal_sy = goal_pos[1] * zoomed_cell_size - camera_y + zoomed_cell_size/2
        
        pygame.draw.circle(screen, colors['blue'], (agent_sx, agent_sy), zoomed_cell_size / 2)
        pygame.draw.circle(screen, colors['green'], (goal_sx, goal_sy), zoomed_cell_size / 2)
        
        # --- UI Overlay ---
        zoom_text = font.render(f"Zoom: {zoom:.2f}x", True, colors['white']); screen.blit(zoom_text, (10, 10))
        controls_text = font.render("ARROWS: Pan | +/-: Zoom", True, colors['white']); screen.blit(controls_text, (10, 40))

        pygame.display.flip()

        if agent_pos == goal_pos:
            print("Goal reached!"); pygame.time.wait(1000); running = False
        clock.tick(speed)

    pygame.quit()

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    number_of_runs = 1
    for i in range(number_of_runs):
        print(f"\n--- Starting Visual Test Run: {i + 1} of {number_of_runs} ---")
        run_visual_test(
            width=1001,  # Must be odd
            height=1001, # Must be odd
            loops=5000,  # Add many loops for a complex maze
            speed=240    # Run at a smooth 240 FPS
        )
    print("\nAll visual tests complete.")
