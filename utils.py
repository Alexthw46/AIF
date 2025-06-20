import math
from collections import deque
from typing import Tuple, List

import numpy as np

directions = ["UP", "RIGHT", "DOWN", "LEFT"]


def get_player_location(game_map: np.ndarray, symbol: str = "@") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return x[0], y[0]


def get_target_location(game_map: np.ndarray, symbol: str = ">") -> Tuple[int, int]:
    x, y = np.where(game_map == ord(symbol))
    return x[0], y[0]


def is_wall(position_element: int) -> bool:
    obstacles = "|-} "
    return chr(position_element) in obstacles


def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int], avoid_stairs=False) -> List[
    Tuple[int, int]]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y - 1]):
        if not (avoid_stairs and game_map[x, y - 1] == ord('>')):
            valid.append((x, y - 1))

    # East
    if x + 1 < x_limit and not is_wall(game_map[x + 1, y]):
        if not (avoid_stairs and game_map[x + 1, y] == ord('>')):
            valid.append((x + 1, y))

    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y + 1]):
        if not (avoid_stairs and game_map[x, y + 1] == ord('>')):
            valid.append((x, y + 1))

    # West
    if x - 1 > 0 and not is_wall(game_map[x - 1, y]):
        if not (avoid_stairs and game_map[x - 1, y] == ord('>')):
            valid.append((x - 1, y))

    return valid


def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> List[int]:
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3
    }
    actions = []
    x_s, y_s = start
    for (x, y) in path:
        if x_s == x:
            if y_s > y:
                actions.append(action_map["W"])
            else:
                actions.append(action_map["E"])
        elif y_s == y:
            if x_s > x:
                actions.append(action_map["N"])
            else:
                actions.append(action_map["S"])
        else:
            # Both x and y change, we split into two actions. But we check that they change by at most 1.
            if abs(x_s - x) > 1 or abs(y_s - y) > 1:
                print(f"Invalid path: {(x_s, y_s)} -> {x, y}")
                raise Exception("Invalid path: x and y can't change by more than 1 at the same time.")
            if x_s > x:
                actions.append(action_map["N"])
            elif x_s < x:
                actions.append(action_map["S"])
            if y_s > y:
                actions.append(action_map["W"])
            elif y_s < y:
                actions.append(action_map["E"])
        x_s = x
        y_s = y

    return actions


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def cached_bfs(game_map, start, goal, path_cache):
    key = (start, goal)
    if key in path_cache:
        return path_cache[key]
    dist = bfs_path_length(game_map, start, goal)
    path_cache[key] = dist
    return dist


def bfs_path_length(game_map, start, goal) -> int:
    """Return shortest path length between start and goal, accounting for walls."""
    if start == goal:
        return 0
    rows, cols = game_map.shape
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        (x, y), dist = queue.popleft()
        for nx, ny in get_valid_moves(game_map, (x, y)):
            if (nx, ny) == goal:
                return dist + 1
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return float('inf')  # no path

def randomize_apple_positions(
        map_str, min_x, min_y, max_x, max_y, num_apple, seed=None
):
    """
    Randomizes apple positions in a given map string.

    Args:
        map_str (str): The map in string representation.
        min_x (int): Minimum x-coordinate for apple placement.
        min_y (int): Minimum y-coordinate for apple placement.
        max_x (int): Maximum x-coordinate for apple placement.
        max_y (int): Maximum y-coordinate for apple placement.
        num_apple (int): Number of apple piles to place.

    Returns:
        list: A list of tuples representing the positions of the apple piles.
    """
    import random

    if seed is not None:
        random.seed(seed)

    apple_positions = []
    lines = map_str.split('\n')
    print(f"Placing {num_apple} apples between ({min_x}, {min_y}) and ({max_x}, {max_y})")
    while len(apple_positions) < num_apple:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        # print(f"Trying to place apple at: ({x}, {y})")
        if (x, y) not in apple_positions:
            if lines[y][x] == '.':
                apple_positions.append((x, y))
                print(f"Placed apple at: ({x}, {y})")
            else:
                continue
                #print(f"Position ({x}, {y}) is not valid for apple placement ({lines[y][x]}).")

    return apple_positions


def print_path_on_map(game_map: np.ndarray, path: List[Tuple[int, int]]):
    path_set = set(path)
    for y in range(game_map.shape[0]):
        row = ""
        for x in range(game_map.shape[1]):
            pos = (y, x)
            char = chr(game_map[y, x])
            if pos == path[0]:
                row += "@"
            elif pos == path[-1] and char == '>':
                row += "!"
            elif pos in path_set:
                if char == '%':
                    row += "A"  # collected apple
                elif char == '>':
                    row += ">"
                else:
                    row += "*"  # part of the path
            else:
                row += char
        print(row)


def simulate_path(path, game_map, actions):
    """ Simulate the path on the game map and print the actions taken.
    """
    print("Actions to take:", list((map(lambda x: directions[x], actions))))
    # check how much apple is collected in the path
    apple_collected = []
    for (x, y) in path:
        if chr(game_map[x, y]) == '%' and (x, y) not in apple_collected:
            apple_collected.append((x, y))
    apple_collected = len(apple_collected)
    print("Apple collected in the path:", apple_collected)
    print("Expected Reward: ", 1 + apple_collected * 0.75 - 0.1 * len(path))
    print_path_on_map(game_map, path)
