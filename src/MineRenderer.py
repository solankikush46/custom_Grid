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
    """
    def __init__(self, n_rows, n_cols):
        """
        Initializes the Pygame window and assets.

        Args:
            n_rows (int): The number of rows in the simulation grid.
            n_cols (int): The number of columns in the simulation grid.
        """
        pygame.init()
        self.n_rows = n_rows
        self.n_cols = n_cols

        # --- Window and Asset Initialization ---
        # Calculate the size of each grid cell to fit the window on the screen
        MAX_WIDTH, MAX_HEIGHT = 1000, 750
        self.cell_size = int(min(MAX_WIDTH / self.n_cols, MAX_HEIGHT / self.n_rows))
        
        # Create the main display surface
        self.screen = pygame.display.set_mode((self.n_cols * self.cell_size, self.n_rows * self.cell_size))
        pygame.display.set_caption("Mine Simulator")
        
        # Initialize the font for drawing text (e.g., battery levels)
        self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))
        
        # The clock is used to control the frame rate
        self.clock = pygame.time.Clock()
        self.render_fps = 4

    def render(self, static_grid, world_state, show_miners=True, dstar_path=None, path_history=None):
        """
        The main rendering loop for a single frame. It calls helper functions
        to draw each component of the scene in a specific order to create layers.

        Args:
            static_grid (np.ndarray): The grid containing permanent terrain obstacles.
            world_state (dict): The dictionary from simulator.get_state_snapshot().
            show_miners (bool): If True, the autonomous miners will be rendered.
            dstar_path (list, optional): The future path calculated by the planner.
            path_history (list, optional): The historical path the miner has traveled.
        """
        # --- The order of these calls determines the layering of the final image ---
        
        # Layer 1: The static background (terrain, goals, grid lines)
        self._draw_base_grid(static_grid, world_state.get('goal_positions', []))
        
        # Layer 2: The historical trail of where the miner has been
        self._draw_path_history(path_history)
        
        # Layer 3: The future path calculated by D* Lite
        self._draw_dstar_path(dstar_path)
        
        # Layer 4: The other "NPC" entities (base stations, autonomous miners)
        self._draw_entities(world_state, show_miners)
        
        # Layer 5: The player-controlled miner (drawn on top of most other things)
        self._draw_guided_miner(world_state.get('guided_miner_pos'))
        
        # Layer 6: Overlays like sensor icons and text (drawn on top of everything)
        self._draw_overlays(world_state.get('sensor_batteries', {}))
        
        # After all drawing is complete, update the display to show the new frame
        pygame.display.flip()
        self.clock.tick(self.render_fps)
        
        # Check for user input (like closing the window) and return status
        return self.handle_events()

    def _draw_base_grid(self, static_grid, goal_positions):
        """Draws the background, static obstacles, goals, and grid lines."""
        self.screen.fill(RENDER_COLORS[EMPTY_CHAR])
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # The logic check uses the efficient Integer ID
                if static_grid[r, c] == OBSTACLE_ID:
                    # The drawing lookup uses the human-readable Character symbol
                    pygame.draw.rect(self.screen, RENDER_COLORS[OBSTACLE_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
                elif (r,c) in goal_positions:
                     pygame.draw.rect(self.screen, RENDER_COLORS[GOAL_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
                
                # Draw a black border for every cell to create the grid effect
                pygame.draw.rect(self.screen, (0, 0, 0), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1)

    def _draw_entities(self, world_state, show_miners):
        """Draws non-controllable entities like base stations and autonomous miners."""
        for r, c in world_state.get('base_station_positions', []):
            pygame.draw.rect(self.screen, RENDER_COLORS[BASE_STATION_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
        
        # Conditionally draw the autonomous miners based on the `show_miners` flag
        if show_miners:
            for r, c in world_state.get('miner_positions', []):
                pygame.draw.rect(self.screen, RENDER_COLORS[MINER_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

    def _draw_path_history(self, path_history):
        """Draws the guided miner's historical trail as a series of connected lines."""
        if path_history and len(path_history) > 1:
            # The list preserves the sequence, so we can draw it as a continuous line
            path_points = [(c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2) for r, c in path_history]
            # Use the trail color defined in constants.py
            pygame.draw.lines(self.screen, RENDER_COLORS["TRAIL"], False, path_points, TRAIL_PATH_THICKNESS)
            
    def _draw_dstar_path(self, path):
        """Draws the future planned path calculated by D* Lite."""
        if path and len(path) > 1:
            # D* Lite path is in (x,y) or (col,row), so the conversion is direct
            path_points = [(x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2) for x, y in path]
            # Draw the lines on the screen
            pygame.draw.lines(self.screen, RENDER_COLORS["DSTAR"], False, path_points, DSTAR_PATH_THICKNESS)
            
    def _draw_guided_miner(self, guided_miner_pos):
        """Draws the single controllable guided miner."""
        if guided_miner_pos:
            r, c = guided_miner_pos
            pygame.draw.rect(self.screen, RENDER_COLORS[GUIDED_MINER_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

    def _draw_overlays(self, sensor_batteries):
        """Draws elements that should appear on top of everything, like sensor icons and text."""
        for (r, c), battery in sensor_batteries.items():
            pygame.draw.rect(self.screen, RENDER_COLORS[SENSOR_CHAR], (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
            # Create a text surface (label) for the battery level
            label = self.font.render(f"{int(battery)}", True, (0, 0, 0)) # Black text
            # Draw the label centered on the sensor's cell
            self.screen.blit(label, label.get_rect(center=(c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)))

    def handle_events(self):
        """Handles user input, like closing the window. Returns False if user quits."""
        for event in pygame.event.get():
            # Check if the user clicked the window's close button or pressed ESC
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return False # Signal to the main loop to stop
        return True # Signal to continue
    
    def close(self):
        """Shuts down the Pygame window properly."""
        pygame.quit()
