import heapq
import math
import pygame
import time
import random

class DStarLite:
    """
    An optimized D* Lite pathfinding algorithm that supports dynamic edge costs
    and multiple goal destinations by initializing all goals as zero-cost path sources.
    """
    def __init__(self, grid, start, goals, cost_function):
        if not goals:
            raise ValueError("Goals list cannot be empty.")

        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.cost_function = cost_function

        # --- Multi-Goal Setup (Simpler/Correct Approach) ---
        self.real_goals = goals
        # s_goal is only needed for the heuristic, can be any goal
        self.s_goal = self.real_goals[0]

        # --- D* Lite Core Variables ---
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
        
        # --- Initialize ALL real goals ---
        for goal in self.real_goals:
            self.rhs[goal] = 0
            heapq.heappush(self.U, (self._calculate_key(goal), goal))
            self.U_members.add(goal)

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
        key, node = heapq.heappop(self.U)
        self.U_members.remove(node)
        return key, node

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
                    self.rhs[s_prime] = min(self.rhs.get(s_prime, float('inf')), self._get_edge_cost(u, s_prime) + self.g.get(u, float('inf')))
                    self._update_node(s_prime)
            else:
                g_old = self.g.get(u, float('inf')); self.g[u] = float('inf')
                for s_prime in self._get_neighbors(u) + [u]:
                    if self.rhs.get(s_prime, float('inf')) == self._get_edge_cost(u, s_prime) + g_old:
                        if s_prime not in self.real_goals:
                            min_rhs = float('inf')
                            for s_hat in self._get_neighbors(s_prime):
                                min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
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
                if s_prime not in self.real_goals:
                    min_rhs = float('inf')
                    for s_hat in self._get_neighbors(s_prime):
                        min_rhs = min(min_rhs, self._get_edge_cost(s_prime, s_hat) + self.g.get(s_hat, float('inf')))
                    self.rhs[s_prime] = min_rhs
                self._update_node(s_prime)
        self._compute_shortest_path()

    def get_path_to_goal(self):
        path = []; curr = self.s_start
        for _ in range(self.width * self.height):
            if curr in self.real_goals:
                path.append(curr); return path
            
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
    # (This helper class for the visual demo remains unchanged)
    def __init__(self, width, height, loops_to_add=0):
        self.width=width; self.height=height; self.dynamic_costs={}; self.impassable_nodes=set(); self.grid=self._generate_maze(); self._add_loops(loops_to_add)
    def _generate_maze(self):
        grid=[[1]*self.width for _ in range(self.height)]; stack=[]; sx,sy=(1,1); grid[sy][sx]=0; stack.append((sx,sy))
        while stack:
            cx,cy=stack[-1]; neighbors=[]
            for dx,dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                nx,ny=cx+dx,cy+dy
                if 0<nx<self.width-1 and 0<ny<self.height-1 and grid[ny][nx]==1: neighbors.append((nx,ny))
            if neighbors: nx,ny=random.choice(neighbors); grid[ny][nx]=0; grid[cy+(ny-cy)//2][cx+(nx-cx)//2]=0; stack.append((nx,ny))
            else: stack.pop()
        return grid
    def _add_loops(self, n):
        walls=[];_=[walls.append((x,y)) for y in range(1,self.height-1) for x in range(1,self.width-1) if self.grid[y][x]==1 and ((self.grid[y][x-1]==0 and self.grid[y][x+1]==0) or (self.grid[y-1][x]==0 and self.grid[y+1][x]==0))]
        random.shuffle(walls);[self._set_grid_val(walls[i],0) for i in range(min(n,len(walls)))]
    def _set_grid_val(self,pos,val): self.grid[pos[1]][pos[0]]=val
    def get_cost(self,pos): return float('inf') if pos in self.impassable_nodes else self.dynamic_costs.get(pos,0)
    def draw(self, screen, cam_x, cam_y, zoom, colors):
        w,h=screen.get_size();zcs=max(1,int(8*zoom));sc=int(cam_x/zcs);ec=sc+int(w/zcs)+2;sr=int(cam_y/zcs);er=sr+int(h/zcs)+2
        for y in range(max(0,sr),min(self.height,er)):
            for x in range(max(0,sc),min(self.width,ec)):
                sx=x*zcs-cam_x;sy=y*zcs-cam_y;rect=pygame.Rect(sx,sy,zcs,zcs)
                if (x,y) in self.impassable_nodes: c=colors['red']
                elif self.grid[y][x]==1: c=colors['black']
                elif (x,y) in self.dynamic_costs: c=colors['cost']
                else: c=colors['white']
                pygame.draw.rect(screen,c,rect)

def run_visual_test(width=101, height=101, loops=50, speed=60):
    pygame.init()
    screen_w, screen_h = 1280, 800
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("D* Lite Multi-Goal: Use ARROWS to Pan, +/- to Zoom")
    clock = pygame.time.Clock()

    colors={'white':(255,255,255),'black':(0,0,0),'blue':(0,0,255),'green':(0,255,0),'red':(255,0,0),'path':(30,144,255),'cost':(255,255,0)}
    font = pygame.font.Font(None, 30)

    loading_font=pygame.font.Font(None,50);text=loading_font.render("Generating Maze & Computing Path...",True,colors['red'])
    text_rect=text.get_rect(center=(screen_w/2,screen_h/2));screen.fill(colors['black']);screen.blit(text,text_rect);pygame.display.flip()

    world=DynamicGridWorld(width,height,loops_to_add=loops)
    start_pos=(1,1)
    goal_positions=[(width-2,height-2),(width-2,1),(1,height-2)]
    agent_pos=start_pos
    
    dstar=DStarLite(world.grid,agent_pos,goal_positions,world.get_cost)
    dstar._compute_shortest_path();path=dstar.get_path_to_goal()

    zoom=2.0;cam_x,cam_y=0,0;pan_speed=20

    running = True
    while running:
        zcs = max(1, int(8 * zoom))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx,my=pygame.mouse.get_pos();wx=int((mx+cam_x)/zcs);wy=int((my+cam_y)/zcs);pos=(wx,wy)
                if 0<=wx<width and 0<=wy<height and world.grid[wy][wx]==0 and pos not in goal_positions:
                    if event.button==3:
                        if pos in world.impassable_nodes: world.impassable_nodes.remove(pos)
                        else: world.impassable_nodes.add(pos)
                    elif event.button==1:
                        if pos in world.dynamic_costs: del world.dynamic_costs[pos]
                        else: world.dynamic_costs[pos]=150.0
                    dstar.update_costs([pos]);path=dstar.get_path_to_goal()
        
        keys=pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_x-=pan_speed
        if keys[pygame.K_RIGHT]: cam_x+=pan_speed
        if keys[pygame.K_UP]: cam_y-=pan_speed
        if keys[pygame.K_DOWN]: cam_y+=pan_speed
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]: zoom*=1.1
        if keys[pygame.K_MINUS]: zoom/=1.1

        if agent_pos not in goal_positions and path and len(path)>1:
            next_pos=path[1];dstar.move_and_replan(next_pos);agent_pos=next_pos;path=dstar.get_path_to_goal()

        cam_x=agent_pos[0]*zcs-screen_w/2;cam_y=agent_pos[1]*zcs-screen_h/2

        screen.fill(colors['black'])
        world.draw(screen,cam_x,cam_y,zoom,colors)

        if path:
            path_points=[(p[0]*zcs-cam_x+zcs/2,p[1]*zcs-cam_y+zcs/2) for p in path]
            if len(path_points)>1: pygame.draw.lines(screen,colors['path'],False,path_points,max(1,int(2*zoom)))
        
        agent_sx,agent_sy=(agent_pos[0]*zcs-cam_x+zcs/2,agent_pos[1]*zcs-cam_y+zcs/2)
        pygame.draw.circle(screen,colors['blue'],(agent_sx,agent_sy),zcs/1.8)
        
        for goal_pos in goal_positions:
            goal_sx,goal_sy=(goal_pos[0]*zcs-cam_x+zcs/2,goal_pos[1]*zcs-cam_y+zcs/2)
            pygame.draw.circle(screen,colors['green'],(goal_sx,goal_sy),zcs/1.8)

        zoom_text=font.render(f"Zoom: {zoom:.2f}x",True,colors['white']);screen.blit(zoom_text,(10,10))
        controls_text=font.render("ARROWS: Pan| +/-: Zoom| Click: Modify",True,colors['white']);screen.blit(controls_text,(10,40))
        pygame.display.flip()

        if agent_pos in goal_positions:
            print("Goal reached!");pygame.time.wait(1000);running=False
        clock.tick(speed)

    pygame.quit()

if __name__ == '__main__':
    run_visual_test(width=501, height=501, loops=500, speed=240)
