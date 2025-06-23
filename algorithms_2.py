import itertools
from queue import PriorityQueue
from typing import Callable
from typing import Set, List, Tuple

import numpy as np

from algorithms import build_path
from utils import get_valid_moves, manhattan_distance


def beam_search_path_planner(game_map: np.ndarray,
                             start: Tuple[int, int],
                             target: Tuple[int, int],
                             apple_positions: Set[Tuple[int, int]],
                             beam_width: int = 3,
                             apple_reward: float = 0.75) -> List[Tuple[int, int]]:
    """
    Beam search to find a path from start to target collecting apples
    to maximize (apple rewards - path cost).

    :param: game_map: 2D grid representing the game state.
    :param: start: Starting coordinate (x, y).
    :param: target: Target coordinate (x, y) to reach.
    :param: apple_positions: Set of coordinates where apples are located.
    :param: beam_width: Maximum number of paths to keep at each step.
    :param: apple_reward: Reward value for each collected apple.
    :return: List of coordinates representing the best path from start to target.
    """

    # Convert apple set to list for indexing
    apple_positions = list(apple_positions)
    all_points = [start] + apple_positions + [target]

    # Precompute all paths and distances
    dist = {}
    paths = {}

    for a, b in itertools.combinations(all_points, 2):
        path = a_star_apple(game_map, a, b, h=manhattan_distance,
                            apple_bonus=apple_reward,
                            apple_positions=set(apple_positions))
        if not path or len(path) < 2:
            dist[(a, b)] = dist[(b, a)] = float('inf')
            paths[(a, b)] = paths[(b, a)] = []
        else:
            d = len(path) - 1
            dist[(a, b)] = dist[(b, a)] = d
            paths[(a, b)] = paths[(b, a)] = path

    # Beam entries: (net reward, total reward, cost, current position, visited apples, full path)
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

            for next_pt in apple_positions + [target]:
                if next_pt in visited and next_pt != target:
                    continue

                path_segment = paths.get((pos, next_pt), [])
                if not path_segment or len(path_segment) < 2:
                    continue

                # Ensure valid connection
                if path_segment[0] != pos:
                    continue

                # Avoid cycles or redundant revisits
                if any(p in path_so_far for p in path_segment[1:-1]):
                    continue

                d = dist.get((pos, next_pt), float('inf'))
                if not np.isfinite(d):
                    continue

                new_reward = reward + (apple_reward if next_pt in apple_positions else 0.0)
                new_cost = cost + d
                new_net = new_reward - new_cost
                new_visited = visited | {next_pt} if next_pt in apple_positions else visited
                new_path = path_so_far + path_segment[1:]
                candidates.append((new_net, new_reward, new_cost, next_pt, new_visited, new_path))

        if not candidates:
            break

        # Take top candidates by net reward
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

        # Update best path if any finished at target
        for net, _, _, pos, _, path in beam:
            if pos == target and net > best_net:
                best_net = net
                best_path = path

        # Only keep non-terminal entries in beam to allow further expansion
        beam = [entry for entry in beam if entry[3] != target]

    print(f"Best net reward: {best_net}, Path length: {len(best_path)}")
    return best_path


def a_star_apple(
        game_map: np.ndarray,
        start: Tuple[int, int],
        target: Tuple[int, int],
        apple_positions: Set[Tuple[int, int]],
        h: Callable[[Tuple[int, int], Tuple[int, int]], float],
        apple_bonus: float = 0.75,
        weight: float = 1.0
) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm that prioritizes paths close to apples ('%'), optional weighted A*.

    :param game_map: 2D grid representing the game state.
    :param start: Starting coordinate (x, y).
    :param target: Target coordinate (x, y) to reach.
    :param apple_positions: Set of coordinates where apples are located.
    :param h: Heuristic function to estimate distance to target.
    :param apple_bonus: Bonus to reduce cost when collecting apples.
    :param weight: Weighting factor for the heuristic.
    :return: List of coordinates representing the path from start to target.

    """

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
                if (nx, ny) in apple_positions:
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
            if neighbor in apple_positions:
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
