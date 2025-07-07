from typing import List, Tuple

import numpy as np

from algorithms import a_star
from utils import get_stairs_location, manhattan_distance, get_valid_moves, bfs_path_length


def a_star_online(game_map, start, **kwargs):
    """
    Online A* algorithm for use with simulate_online.
    Plans a path to the nearest apple, or to the stairs if no apples remain.
    More similar to the a_star_collect_apples function, but adapted for online use.
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


def planner_online(game_map, start, planner_func, verbose=True, **kwargs):
    """
    General online planner function for use with simulate_online.
    Takes a planning function (like mcts or greedy_best_first_online) as an argument.
    Plans a path to the nearest apple, or to the stairs if no apples remain.
    """
    apple_positions, target = find_target(game_map, start, verbose=verbose)
    if target is None:
        print("No target found, returning empty.")
        return []
    if verbose:
        print("Apple positions:", apple_positions)
        print("Target:", target)
    return planner_func(game_map, start, target, set(apple_positions), **kwargs)


def score_frontier(game_map: np.ndarray, start: Tuple[int, int], frontier_cell: Tuple[int, int], info_gain=0) -> float:
    """
    Score the frontier tiles based on their distance from the start position and number of adjacent unknown tiles.
    """
    path_len = bfs_path_length(game_map, start, frontier_cell)
    if path_len == float('inf') or path_len == 0:
        return -float('inf')  # Skip unreachable or current position

    # Count unknown tiles adjacent to the frontier cell
    for ny, nx in get_valid_moves(game_map, frontier_cell):
        if game_map[ny, nx] == ord(' '):  # Unknown tile
            info_gain += 1

    return info_gain / path_len


def find_target(game_map, start, verbose=True) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
        Determines the next target for the agent to pursue.

        Finds all apple positions and the stairs location. If stairs are not found, evaluates frontier tiles
        (known tiles adjacent to unknown space) and scores them for information gain. If apples are available,
        scores them as well and chooses between the best apple and the best frontier tile. Returns the list of
        apple positions and the chosen target position.

        Args:
            game_map: 2D numpy array representing the map.
            start: Tuple of (y, x) for the agent's current position.
            verbose: If True, prints debug information.

        Returns:
            Tuple containing:
                - List of (y, x) tuples for apple positions.
                - (y, x) tuple for the chosen target, or None if no target is found.
        """
    if verbose: print("Finding target from start:", start)
    char_map = np.vectorize(chr)(game_map)
    apple_positions = np.where(char_map == '%')
    apple_positions = list(zip(apple_positions[0], apple_positions[1]))

    # Stairs are the primary target
    target = get_stairs_location(game_map)

    # If stairs are not found, decide based on apples or frontier
    if target is None:
        if verbose: print("No stairs found, evaluating frontier.")
        frontier = frontier_search(game_map)
        best_score = -float('inf')
        # Score the frontier tiles to find the one with the best information gain
        for pos in frontier:
            score = score_frontier(game_map, start, pos)
            if score > best_score:
                best_score = score
                target = pos

        if verbose:
            if target is not None:
                print("Using frontier point as target:", target)
            else:
                print("No frontier found, whole map explored?")

        # If apples are available, consider them as well
        if len(apple_positions) > 0:
            best_apple = None
            best_apple_score = -float('inf')
            for apple in apple_positions:
                score = score_frontier(game_map, start, apple, info_gain=1)  # Apples give a fixed info gain of 1
                if score > best_apple_score:
                    best_apple_score = score
                    best_apple = apple

            # Decide whether to target an apple or the frontier, if both are available
            if target is not None:
                if best_apple_score > score_frontier(game_map, start, target):
                    target = best_apple
                    if verbose: print("Targeting apple instead of frontier:", target)
            else:
                target = best_apple
                if verbose: print("Targeting closest apple as no frontier found:", target)

    if target is None:
        if verbose: print("No target found, returning empty.")

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
            # Check if the current tile is unknown
            if game_map[y, x] == ord(' '):
                # Check moore neighborhood for walkable tiles
                for ny, nx in get_valid_moves(game_map, (y, x)):
                    # Check if the adjacent tile is known
                    if game_map[ny, nx] != ord(' '):
                        frontier.append((y, x))
                        break

    return frontier
