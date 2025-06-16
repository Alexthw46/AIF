import heapq
import itertools
from queue import PriorityQueue
from typing import Tuple, List, Callable

import numpy as np
from algorithms import build_path
from utils import get_valid_moves, manhattan_distance


def beam_search_path_planner(game_map: np.ndarray,
                             start: Tuple[int, int],
                             target: Tuple[int, int],
                             apples: List[Tuple[int, int]],
                             beam_width: int = 3,
                             apple_reward: float = 0.75) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Finds a path from start to target collecting apples to maximize net reward (rewards - path cost).

    Parameters:
    - game_map: 2D grid
    - start: starting coordinate
    - target: target coordinate (e.g., stairs)
    - apples: list of apple coordinates
    - beam_width: number of top partial paths to keep per iteration
    - apple_reward: reward for collecting each apple

    Returns:
    - best_net: maximum net reward achieved
    - best_path: sequence of grid coordinates from start to target
    """
    # All key points: apples and target
    all_points = [start] + apples + [target]
    dist = {}
    paths = {}

    for a, b in itertools.combinations(all_points, 2):
        path = a_star_apple(game_map, a, b, h=manhattan_distance, apple_bonus=apple_reward)
        if path is None:
            dist[(a, b)] = dist[(b, a)] = float('inf')
            paths[(a, b)] = paths[(b, a)] = []
        else:
            d = len(path) - 1
            dist[(a, b)] = dist[(b, a)] = d
            paths[(a, b)] = paths[(b, a)] = path

    beam = [(0.0, 0.0, 0.0, start, frozenset(), [start])]
    best_net = float('-inf')
    best_path: List[Tuple[int, int]] = []

    while beam:
        candidates = []
        for net, reward, cost, pos, visited, path_so_far in beam:
            if pos == target:
                if net > best_net:
                    best_net = net
                    best_path = path_so_far
                continue

            for next_pt in apples + [target]:
                if next_pt in visited and next_pt != target:
                    continue
                d = dist.get((pos, next_pt), float('inf'))
                if not np.isfinite(d):
                    continue

                new_reward = reward + (apple_reward if next_pt in apples else 0.0)
                new_cost = cost + d
                new_net = new_reward - new_cost
                new_visited = visited | {next_pt} if next_pt in apples else visited
                new_path = path_so_far + paths[(pos, next_pt)][1:]
                candidates.append((new_net, new_reward, new_cost, next_pt, new_visited, new_path))

        if not candidates:
            break

        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

        for net, _, _, pos, _, path in beam:
            if pos == target and net > best_net:
                best_net = net
                best_path = path

        beam = [entry for entry in beam if entry[3] != target]

    return best_net, best_path


def a_star_apple(
        game_map: np.ndarray,
        start: Tuple[int, int],
        target: Tuple[int, int],
        h: Callable[[Tuple[int, int], Tuple[int, int]], float],
        apple_bonus: float = 0.75,
        weight: float = 1.0
) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm that prioritizes paths close to apples ('%'), optional weighted A*.

    Parameters:
        game_map (np.ndarray): 2D grid map with cells as ASCII codes.
        start (Tuple[int, int]): Starting coordinates.
        target (Tuple[int, int]): Target coordinates.
        h (Callable): Heuristic function estimating cost from node to target.
        apple_bonus (float): Cost reduction applied for proximity to apples.
        weight (float): Weight for the heuristic.
    Returns:
        List[Tuple[int, int]]: The path from start to target, empty if none found.
    """

    APPLE = ord('%')  # ASCII code for apple character
    rows, cols = game_map.shape

    # Priority queue stores nodes with their f-score = g + h
    open_list = PriorityQueue()

    # g_scores: cost from start to current node
    g_scores = {start: 0}

    # parent dictionary to reconstruct path
    parent = {start: None}

    # Set of nodes already evaluated
    closed_set = set()

    def apple_in_vicinity(pos: Tuple[int, int]) -> bool:
        """
        Check if there is at least one apple in the 8 adjacent cells around pos.
        """
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # skip the current cell itself
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if game_map[nx, ny] == APPLE:
                        return True
        return False

    # Initial node f-score = heuristic only since g=0
    f_start = h(start, target)
    open_list.put((f_start, start))

    while not open_list.empty():
        # Get node with lowest f-score
        _, current = open_list.get()

        # Skip if already evaluated
        if current in closed_set:
            continue

        # Mark current node as evaluated
        closed_set.add(current)

        # Check if target reached; reconstruct path
        if current == target:
            return build_path(parent, target)

        current_g = g_scores[current]

        # Explore neighbors of current node
        for neighbor in get_valid_moves(game_map, current):
            if neighbor in closed_set:
                continue  # skip neighbors already evaluated

            # Base cost to move from current to neighbor (assume uniform cost 1)
            tentative_g = current_g + 1

            # If neighbor is an apple, subtract stronger bonus (reduce cost)
            if game_map[neighbor] == APPLE:
                tentative_g -= apple_bonus * 1.5

            # If apple is near neighbor, apply smaller bonus
            elif apple_in_vicinity(neighbor):
                tentative_g -= apple_bonus * 0.75

            # If neighbor not visited before or found better path
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                parent[neighbor] = current
                f = tentative_g + h(neighbor, target) * weight
                open_list.put((f, neighbor))

    # No path found
    return []

import heapq

def weighted_a_star(start, goal, get_neighbors, heuristic, weight=1.5):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1  # or cost(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + weight * heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current
    return None