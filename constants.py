# constants.py

# grid symbols
EMPTY = '.'
OBSTACLE = '#'
GOAL = 'G'
SENSOR = 'S'
AGENT = 'A'
FINISHED = 'F'
TRAIL_OUTSIDE = '*'
TRAIL_INSIDE = 'T'
RADAR_BG = 'BG'

# rgb colors for pygame rendering
RENDER_COLORS = {
    EMPTY: (255, 255, 255),
    OBSTACLE: (100, 100, 100),
    TRAIL_OUTSIDE: (255, 255, 0),
    AGENT: (0, 0, 255),
    GOAL: (0, 255, 0),
    FINISHED: (0, 255, 255),
    SENSOR: (255, 0, 0),
    TRAIL_INSIDE: (173, 216, 230),
    RADAR_BG: (255, 165, 0),
}

# actions agent can take (cardinal directions and diagonals)
DIRECTION_MAP = {
    0: (-1,  0), 1: (-1, +1), 2: (0, +1), 3: (+1, +1),
    4: (+1,  0), 5: (+1, -1), 6: (0, -1), 7: (-1, -1),
}

