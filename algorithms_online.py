import numpy as np

from algorithms import a_star
from utils import get_target_location, manhattan_distance


def a_star_online(game_map, start, **kwargs):
    """
    Online A* algorithm for use with simulate_online.
    Plans a path to the nearest apple, or to the stairs if no apples remain.
    """
    char_map = np.vectorize(chr)(game_map)
    apple_positions = np.where(char_map == '%')
    apple_positions = list(zip(apple_positions[0], apple_positions[1]))

    if apple_positions:
        # Find the closest apple
        targets = apple_positions
    else:
        # No apples left, go to stairs
        stairs = get_target_location(game_map)
        if stairs is None:
            return []
        targets = [stairs]

    # Find the closest target
    min_path = None
    min_len = float('inf')
    for target in targets:
        path = a_star(game_map, start, target, manhattan_distance)
        if path and len(path) < min_len:
            min_path = path
            min_len = len(path)
    return min_path if min_path else []