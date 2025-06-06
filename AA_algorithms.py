import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path

def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    # initialize open and close list
    open_list = PriorityQueue()
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = 0
    starting_state_h = h(start, target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        # get the node with lowest f
        _, (current, current_cost) = open_list.get()
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current):
            # check if neighbor in close list, if so continue
            if neighbor in close_list:
                continue
            # compute neighbor g, h and f values
            neighbor_g = 1 + current_cost
            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h
            parent[neighbor] = current
            neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                if neighbor_g >= support_list[neighbor]:
                    continue
            
            # add neighbor to open list and update support_list
            open_list.put(neighbor_entry)
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return []

def a_star_gold(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable, gold_bonus: float = 1.5) -> List[Tuple[int, int]]:
    from queue import PriorityQueue

    open_list = PriorityQueue()
    closed_list = set()
    support_list = {}
    parent = {}

    starting_state_g = 0
    starting_state_h = h(start, target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent[start] = None

    while not open_list.empty():
        _, (current, current_cost) = open_list.get()
        if current in closed_list:
            continue
        closed_list.add(current)

        if current == target:
            return build_path(parent, target)

        for neighbor in get_valid_moves(game_map, current):
            if neighbor in closed_list:
                continue

            # Base movement cost
            neighbor_g = current_cost + 1

            # Bonus for standing on gold
            if game_map[neighbor] == ord('$'):
                neighbor_g -= gold_bonus  # more attractive
            # Bonus for being adjacent to gold
            else:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = neighbor[0] + dx, neighbor[1] + dy
                        if 0 <= nx < game_map.shape[0] and 0 <= ny < game_map.shape[1]:
                            if game_map[nx, ny] == ord('$'):
                                neighbor_g -= gold_bonus / 2  # less than standing on it
                                break

            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h

            if neighbor in support_list and neighbor_g >= support_list[neighbor]:
                continue

            parent[neighbor] = current
            support_list[neighbor] = neighbor_g
            open_list.put((neighbor_f, (neighbor, neighbor_g)))

    return []
