# MineRenderer.py

import pygame
from .constants import *

class MineRenderer:
    def __init__(self, n_rows, n_cols):
        pygame.init()
        self.n_rows = n_rows
        self.n_cols = n_cols
        MAX_WIDTH, MAX_HEIGHT = 1000, 750
        self.cell_size = int(min(MAX_WIDTH / self.n_cols, MAX_HEIGHT / self.n_rows))
        self.screen = pygame.display.set_mode((self.n_cols * self.cell_size, self.n_rows * self.cell_size))
        pygame.display.set_caption("Mine Simulator")
        self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))
        self.clock = pygame.time.Clock()
        self.render_fps = 10

    def render(self, static_grid, world_state, show_miners=False):
        self._draw_base_grid(static_grid, world_state.get('goal_positions', []))
        self._draw_entities(world_state, show_miners=show_miners)
        self._draw_guided_miner(world_state.get('guided_miner_pos'))
        self._draw_overlays(world_state.get('sensor_batteries', {}))
        pygame.display.flip()
        self.clock.tick(self.render_fps)
        return self.handle_events() # Return status from event handler

    def _draw_base_grid(self, static_grid, goal_positions):
        self.screen.fill(RENDER_COLORS[EMPTY_CHAR])
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # Logic uses the Integer ID
                if static_grid[r, c] == OBSTACLE_ID:
                    # Drawing uses the Character symbol to look up color
                    pygame.draw.rect(self.screen, RENDER_COLORS[OBSTACLE_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
                elif (r,c) in goal_positions:
                     pygame.draw.rect(self.screen, RENDER_COLORS[GOAL_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1)

    def _draw_entities(self, world_state, show_miners):
        for r, c in world_state.get('base_station_positions', []):
            pygame.draw.rect(self.screen, RENDER_COLORS[BASE_STATION_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
        for r, c in world_state.get('miner_positions', []):
            pygame.draw.rect(self.screen, RENDER_COLORS[MINER_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

    def _draw_guided_miner(self, guided_miner_pos):
        if guided_miner_pos:
            r, c = guided_miner_pos
            pygame.draw.rect(self.screen, RENDER_COLORS[GUIDED_MINER_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

    def _draw_overlays(self, sensor_batteries):
        for (r, c), battery in sensor_batteries.items():
            pygame.draw.rect(self.screen, RENDER_COLORS[SENSOR_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
            label = self.font.render(f"{int(battery)}", True, (0, 0, 0))
            self.screen.blit(label, label.get_rect(center=(c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return False
        return True
    
    def close(self):
        pygame.quit()
