from typing import List, Tuple

import numpy as np

from algorithms import a_star
from utils import get_stairs_location, manhattan_distance, get_valid_moves, bfs_path_length


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
        stairs = get_stairs_location(game_map)
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


def planner_online(game_map, start, planner_func, **kwargs):
    """
    General online planner function for use with simulate_online.
    Takes a planning function (like mcts or greedy_best_first_online) as an argument.
    Plans a path to the nearest apple, or to the stairs if no apples remain.
    """
    apple_positions, target = find_target(game_map, start)
    print("Apple positions:", apple_positions)
    print("Target:", target)
    return planner_func(game_map, start, target, set(apple_positions), **kwargs)


def find_target(game_map, start):
    print("Finding target from start:", start)
    char_map = np.vectorize(chr)(game_map)
    apple_positions = np.where(char_map == '%')
    apple_positions = list(zip(apple_positions[0], apple_positions[1]))

    target = get_stairs_location(game_map)

    if target is None:
        print("No stairs found, searching for frontier.")
        frontier = frontier_search(game_map)
        # select the closest frontier point as the target
        min_dist = float('inf')
        for pos in frontier:
            dist = bfs_path_length(game_map, start, pos)
            if dist < min_dist:
                min_dist = dist
                target = pos

        if target is not None:
            print("Using frontier point as target:", target)
        else:
            print("No frontier found, whole map explored?")

    if target is None and len(apple_positions) > 0:
        target = apple_positions[0]  # Fallback to the first apple if no stairs or frontier found

    if target is None:
        print("No target found, returning empty.")
        return [], None

    return apple_positions, target


def frontier_search(game_map: np.ndarray) -> List[Tuple[int, int]]:
    """
        Find walkable frontier tiles — known tiles adjacent to unknown space.

        Args:
            game_map: 2D numpy array of characters

        Returns:
            List of (y, x) tuples — frontier tile positions
    """
    frontier = []
    rows, cols = game_map.shape

    for y in range(rows):
        for x in range(cols):
            if game_map[y, x] == ord(' '):
                for ny, nx in get_valid_moves(game_map, (y, x)):
                    if game_map[ny, nx] != ord(' '):
                        frontier.append((y, x))
                        break

    return frontier
