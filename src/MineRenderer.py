# MineRenderer.py

import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL

from .constants import (
    OBSTACLE_ID,
    RENDER_COLORS,
    DSTAR_PATH_THICKNESS,
    TRAIL_PATH_THICKNESS,
    OBSTACLE_CHAR,
    GOAL_CHAR,
    BASE_STATION_CHAR,
    MINER_CHAR,
    GUIDED_MINER_CHAR,
    SENSOR_CHAR
)


class Camera:
    """
    Simple camera for panning and zooming over a large world.
    World coordinates in pixels, origin (0,0) at top-left of grid.
    """
    def __init__(self, viewport_width, viewport_height, world_width, world_height):
        self.vw = viewport_width
        self.vh = viewport_height
        self.ww = world_width
        self.wh = world_height
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def world_to_screen(self, wx, wy):
        sx = (wx - self.offset_x) * self.zoom
        sy = (wy - self.offset_y) * self.zoom
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        wx = sx / self.zoom + self.offset_x
        wy = sy / self.zoom + self.offset_y
        return wx, wy

    def move(self, dx, dy):
        self.offset_x = max(0, min(self.offset_x + dx, self.ww - self.vw / self.zoom))
        self.offset_y = max(0, min(self.offset_y + dy, self.wh - self.vh / self.zoom))

    def change_zoom(self, factor, center_sx, center_sy):
        wx, wy = self.screen_to_world(center_sx, center_sy)
        new_zoom = max(0.1, min(self.zoom * factor, 10.0))
        self.offset_x = wx - center_sx / new_zoom
        self.offset_y = wy - center_sy / new_zoom
        self.zoom = new_zoom
        self.move(0, 0)


class MineRenderer:
    """
    View for the mine simulation, with pan & zoom support.
    """
    def __init__(self, n_rows, n_cols, show_predicted=False):
        pygame.init()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.show_predicted = show_predicted

        MAX_WIDTH, MAX_HEIGHT = 1000, 750
        raw_size = min(MAX_WIDTH / n_cols, MAX_HEIGHT / n_rows)
        self.base_cell_size = max(1, int(raw_size))
        
        base_size = max(10, self.base_cell_size // 3)
        self.font = pygame.font.SysFont("Arial", base_size)
        self.small_font = pygame.font.SysFont("Arial", max(12, base_size // 2))

        world_w = n_cols * self.base_cell_size
        world_h = n_rows * self.base_cell_size
        screen_w = min(world_w, MAX_WIDTH)
        screen_h = min(world_h, MAX_HEIGHT)
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Mine Simulator")

        sw, sh = self.screen.get_size()
        self.camera = Camera(sw, sh, world_w, world_h)

        self.clock = pygame.time.Clock()
        self.render_fps = 60
        self.dragging = False
        self.last_mouse = (0, 0)

    def render(self, static_grid, world_state,
               show_miners=True, dstar_path=None,
               path_history=None, predicted_battery_map=None):
        if not self.handle_events():
            return False

        self.screen.fill((255, 255, 255))
        sw, sh = self.screen.get_size()
        invz = 1.0 / self.camera.zoom
        wx0, wy0 = self.camera.offset_x, self.camera.offset_y
        wx1, wy1 = wx0 + sw * invz, wy0 + sh * invz
        c0 = max(0, int(wx0 // self.base_cell_size))
        r0 = max(0, int(wy0 // self.base_cell_size))
        c1 = min(self.n_cols, int(wx1 // self.base_cell_size) + 1)
        r1 = min(self.n_rows, int(wy1 // self.base_cell_size) + 1)

        self._draw_static_grid(static_grid, world_state, c0, r0, c1, r1)
        if self.show_predicted and predicted_battery_map is not None:
            self._draw_predicted(predicted_battery_map, c0, r0, c1, r1)
        if path_history and len(path_history) > 1:
            self._draw_path_history(path_history)
        if dstar_path and len(dstar_path) > 1:
            self._draw_dstar_path(dstar_path)
        if show_miners:
            self._draw_miners(world_state, c0, r0, c1, r1)
        self._draw_base_stations(world_state, c0, r0, c1, r1)
        self._draw_guided_miner(world_state, c0, r0, c1, r1)
        self._draw_sensor_overlays(world_state, c0, r0, c1, r1)

        pygame.display.flip()
        self.clock.tick(self.render_fps)
        return True

    def _draw_static_grid(self, static_grid, world_state, c0, r0, c1, r1):
        for r in range(r0, r1):
            for c in range(c0, c1):
                xw, yw = c * self.base_cell_size, r * self.base_cell_size
                sx, sy = self.camera.world_to_screen(xw, yw)
                size = int(self.base_cell_size * self.camera.zoom)
                color = None
                if static_grid[r, c] == OBSTACLE_ID:
                    color = RENDER_COLORS[OBSTACLE_CHAR]
                elif (r, c) in world_state.get('goal_positions', []):
                    color = RENDER_COLORS[GOAL_CHAR]
                if color:
                    pygame.draw.rect(self.screen, color, (sx, sy, size, size))
                pygame.draw.rect(self.screen, (0, 0, 0), (sx, sy, size, size), 1)

    def _draw_predicted(self, batt_map, c0, r0, c1, r1):
        for r in range(r0, r1):
            for c in range(c0, c1):
                try:
                    val = int(batt_map[r][c])
                except Exception:
                    continue
                xw = c * self.base_cell_size + self.base_cell_size / 2
                yw = r * self.base_cell_size + self.base_cell_size / 2
                sx, sy = self.camera.world_to_screen(xw, yw)
                label = self.small_font.render(str(val), True, (0, 0, 0))
                self.screen.blit(label, label.get_rect(center=(sx, sy)))

    def _draw_path_history(self, path_history):
        pts = []
        for r, c in path_history:
            xw = c * self.base_cell_size + self.base_cell_size / 2
            yw = r * self.base_cell_size + self.base_cell_size / 2
            pts.append(self.camera.world_to_screen(xw, yw))
        pygame.draw.lines(self.screen, RENDER_COLORS['TRAIL'], False, pts, TRAIL_PATH_THICKNESS)

    def _draw_dstar_path(self, dstar_path):
        pts = []
        for x, y in dstar_path:
            xw = x * self.base_cell_size + self.base_cell_size / 2
            yw = y * self.base_cell_size + self.base_cell_size / 2
            pts.append(self.camera.world_to_screen(xw, yw))
        pygame.draw.lines(self.screen, RENDER_COLORS['DSTAR'], False, pts, DSTAR_PATH_THICKNESS)

    def _draw_miners(self, world_state, c0, r0, c1, r1):
        for r, c in world_state.get('miner_positions', []):
            if r0 <= r < r1 and c0 <= c < c1:
                xw, yw = c * self.base_cell_size, r * self.base_cell_size
                sx, sy = self.camera.world_to_screen(xw, yw)
                size = int(self.base_cell_size * self.camera.zoom)
                pygame.draw.rect(self.screen, RENDER_COLORS[MINER_CHAR], (sx, sy, size, size))

    def _draw_base_stations(self, world_state, c0, r0, c1, r1):
        for r, c in world_state.get('base_station_positions', []):
            if r0 <= r < r1 and c0 <= c < c1:
                xw, yw = c * self.base_cell_size, r * self.base_cell_size
                sx, sy = self.camera.world_to_screen(xw, yw)
                size = int(self.base_cell_size * self.camera.zoom)
                pygame.draw.rect(self.screen, RENDER_COLORS[BASE_STATION_CHAR], (sx, sy, size, size))

    def _draw_guided_miner(self, world_state, c0, r0, c1, r1):
        gp = world_state.get('guided_miner_pos')
        if not gp:
            return
        r, c = gp
        if r0 <= r < r1 and c0 <= c < c1:
            xw, yw = c * self.base_cell_size, r * self.base_cell_size
            sx, sy = self.camera.world_to_screen(xw, yw)
            size = int(self.base_cell_size * self.camera.zoom)
            pygame.draw.rect(self.screen, RENDER_COLORS[GUIDED_MINER_CHAR], (sx, sy, size, size))

    def _draw_sensor_overlays(self, world_state, c0, r0, c1, r1):
        for (r, c), batt in world_state.get('sensor_batteries', {}).items():
            if r0 <= r < r1 and c0 <= c < c1:
                xw, yw = c * self.base_cell_size, r * self.base_cell_size
                sx, sy = self.camera.world_to_screen(xw, yw)
                size = int(self.base_cell_size * self.camera.zoom)
                pygame.draw.rect(self.screen, RENDER_COLORS[SENSOR_CHAR], (sx, sy, size, size))
                label = self.small_font.render(str(int(batt)), True, (0, 0, 0))
                self.screen.blit(label, label.get_rect(center=(sx + size/2, sy + size/2)))

    def handle_events(self):
        """Process Pygame events: pan with drag, zoom with wheel."""
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return False
            elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                self.dragging = True
                self.last_mouse = ev.pos
            elif ev.type == MOUSEBUTTONUP and ev.button == 1:
                self.dragging = False
            elif ev.type == MOUSEMOTION and self.dragging:
                dx = -(ev.pos[0] - self.last_mouse[0]) / self.camera.zoom
                dy = -(ev.pos[1] - self.last_mouse[1]) / self.camera.zoom
                self.camera.move(dx, dy)
                self.last_mouse = ev.pos
            elif ev.type == MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                factor = 1.1 if ev.y > 0 else 0.9
                self.camera.change_zoom(factor, mx, my)
        return True

    def close(self):
        """Shuts down the Pygame window and quits."""
        pygame.quit()
