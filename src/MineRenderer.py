# MineRenderer.py

import pygame

from .constants import (
    EMPTY_CHAR, OBSTACLE_CHAR, SENSOR_CHAR, GUIDED_MINER_CHAR,
    BASE_STATION_CHAR, MINER_CHAR, GOAL_CHAR,
    OBSTACLE_ID,
    RENDER_COLORS,
    DSTAR_PATH_THICKNESS,
    TRAIL_PATH_THICKNESS
)

class MineRenderer:
    """
    Handles all Pygame-related visualization for the mine simulation.
    This class is the "View" in a Model-View-Controller pattern. It is given
    the current state of the world and is responsible for drawing it to the screen.
    It can optionally overlay predicted battery levels per cell.
    """
    def __init__(self, n_rows, n_cols, show_predicted=False):
        """
        Initializes the Pygame window and assets.

        Args:
            n_rows (int): The number of rows in the simulation grid.
            n_cols (int): The number of columns in the simulation grid.
            show_predicted (bool): If True, overlay predicted battery levels per cell.
        """
        pygame.init()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.show_predicted = show_predicted

        # --- Calculate cell size and font ---
        MAX_WIDTH, MAX_HEIGHT = 1000, 750
        self.cell_size = int(min(MAX_WIDTH / self.n_cols, MAX_HEIGHT / self.n_rows))
        self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))

        # --- Margins for indices ---
        self.margin_left = self.font.get_height() + 4
        self.margin_top = self.font.get_height() + 4

        # Create the main display surface with margins
        total_width = self.margin_left + self.n_cols * self.cell_size
        total_height = self.margin_top + self.n_rows * self.cell_size
        self.screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("Mine Simulator")

        # The clock is used to control the frame rate
        self.clock = pygame.time.Clock()
        self.render_fps = 60

    def render(self, static_grid, world_state, show_miners=True,
               dstar_path=None, path_history=None, predicted_battery_map=None):
        """
        Render a single frame, drawing indices outside the grid.
        """
        self.screen.fill((255, 255, 255))
        # Draw indices first so they appear under the grid lines
        self._draw_indices()
        # Draw grid and overlays with offset
        self._draw_base_grid(static_grid, world_state.get('goal_positions', []))
        if self.show_predicted and predicted_battery_map is not None:
            self._draw_predicted(predicted_battery_map)
        self._draw_path_history(path_history)
        self._draw_dstar_path(dstar_path)
        self._draw_entities(world_state, show_miners)
        self._draw_guided_miner(world_state.get('guided_miner_pos'))
        self._draw_overlays(world_state.get('sensor_batteries', {}))
        pygame.display.flip()
        self.clock.tick(self.render_fps)
        return self.handle_events()

    def _draw_indices(self):
        """Draw row labels on left margin and column labels on top margin."""
        # Column indices
        for c in range(self.n_cols):
            label = self.font.render(str(c), True, (0, 0, 0))
            x = self.margin_left + c * self.cell_size + self.cell_size // 2
            y = self.margin_top // 2
            self.screen.blit(label, label.get_rect(center=(x, y)))
        # Row indices
        for r in range(self.n_rows):
            label = self.font.render(str(r), True, (0, 0, 0))
            x = self.margin_left // 2
            y = self.margin_top + r * self.cell_size + self.cell_size // 2
            self.screen.blit(label, label.get_rect(center=(x, y)))

    def _draw_base_grid(self, static_grid, goal_positions):
        """Draws obstacles, goals, and grid lines with offset."""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                x0 = self.margin_left + c * self.cell_size
                y0 = self.margin_top + r * self.cell_size
                if static_grid[r, c] == OBSTACLE_ID:
                    pygame.draw.rect(
                        self.screen,
                        RENDER_COLORS[OBSTACLE_CHAR],
                        (x0, y0, self.cell_size, self.cell_size)
                    )
                elif (r, c) in goal_positions:
                    pygame.draw.rect(
                        self.screen,
                        RENDER_COLORS[GOAL_CHAR],
                        (x0, y0, self.cell_size, self.cell_size)
                    )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),
                    (x0, y0, self.cell_size, self.cell_size),
                    1
                )

    def _draw_predicted(self, batt_map):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                try:
                    txt = f"{int(batt_map[r][c])}"
                except Exception:
                    continue
                label = self.font.render(txt, True, (0, 0, 0))
                cx = self.margin_left + c * self.cell_size + self.cell_size // 2
                cy = self.margin_top + r * self.cell_size + self.cell_size // 2
                self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def _draw_entities(self, world_state, show_miners):
        for r, c in world_state.get('base_station_positions', []):
            x0 = self.margin_left + c * self.cell_size
            y0 = self.margin_top + r * self.cell_size
            pygame.draw.rect(
                self.screen,
                RENDER_COLORS[BASE_STATION_CHAR],
                (x0, y0, self.cell_size, self.cell_size)
            )
        if show_miners:
            for r, c in world_state.get('miner_positions', []):
                x0 = self.margin_left + c * self.cell_size
                y0 = self.margin_top + r * self.cell_size
                pygame.draw.rect(
                    self.screen,
                    RENDER_COLORS[MINER_CHAR],
                    (x0, y0, self.cell_size, self.cell_size)
                )

    def _draw_path_history(self, path_history):
        if path_history and len(path_history) > 1:
            pts = [
                (self.margin_left + c * self.cell_size + self.cell_size // 2,
                 self.margin_top  + r * self.cell_size + self.cell_size // 2)
                for r, c in path_history
            ]
            pygame.draw.lines(
                self.screen,
                RENDER_COLORS["TRAIL"],
                False,
                pts,
                TRAIL_PATH_THICKNESS
            )

    def _draw_dstar_path(self, path):
        if path and len(path) > 1:
            pts = [
                (self.margin_left + x * self.cell_size + self.cell_size // 2,
                 self.margin_top  + y * self.cell_size + self.cell_size // 2)
                for x, y in path
            ]
            pygame.draw.lines(
                self.screen,
                RENDER_COLORS["DSTAR"],
                False,
                pts,
                DSTAR_PATH_THICKNESS
            )

    def _draw_guided_miner(self, guided_miner_pos):
        if guided_miner_pos:
            r, c = guided_miner_pos
            x0 = self.margin_left + c * self.cell_size
            y0 = self.margin_top  + r * self.cell_size
            pygame.draw.rect(
                self.screen,
                RENDER_COLORS[GUIDED_MINER_CHAR],
                (x0, y0, self.cell_size, self.cell_size)
            )

    def _draw_overlays(self, sensor_batteries):
        for (r, c), battery in sensor_batteries.items():
            x0 = self.margin_left + c * self.cell_size
            y0 = self.margin_top  + r * self.cell_size
            pygame.draw.rect(
                self.screen,
                RENDER_COLORS[SENSOR_CHAR],
                (x0, y0, self.cell_size, self.cell_size)
            )
            label = self.font.render(f"{int(battery)}", True, (0, 0, 0))
            cx = x0 + self.cell_size // 2
            cy = y0 + self.cell_size // 2
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return False
        return True

    def close(self):
        pygame.quit()
