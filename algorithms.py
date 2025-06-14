import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List, Set
from utils import *

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path

def bfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Create a queue for BFS and mark the start node as visited
    queue = deque()
    visited = set()
    queue.append(start)
    visited.add(start)

    # Create a dictionary to keep track of the parent node for each node in the path
    parent = {start: None}

    while queue:
        # Dequeue a vertex from the queue
        current = queue.popleft()

        # Check if the target node has been reached
        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        # Visit all adjacent neighbors of the dequeued vertex
        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    print("Target node not found!")
    return None

# ---------------------------------------------

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
    return None

def find_path_with_apples(game_map, start, apples, target, h):
    path = []
    current = start
    apples = apples.copy()
    
    while apples:
        # Find closest apple
        apples.sort(key=lambda apple: h(current, apple))
        next_apple = apples.pop(0)
        
        subpath = a_star(game_map, current, next_apple, h)
        if not subpath:
            return None
        
        if path:
            path += subpath[1:]  # avoid duplicating current
        else:
            path += subpath
            
        current = next_apple

    # Finally go to target
    subpath = a_star(game_map, current, target, h)
    if not subpath:
        return None

    path += subpath[1:]  # avoid duplicate position
    return path

def heuristic_with_apples(current: Tuple[int, int], apples: Set[Tuple[int, int]], target: Tuple[int, int]) -> int:
    remaining = list(apples)
    if not remaining:
        return manhattan_distance(current, target)
    
    # Nearest apple + apple to target (greedy approximation)
    to_apples = [manhattan_distance(current, apple) for apple in remaining]
    from_apples_to_target = [manhattan_distance(apple, target) for apple in remaining]
    
    return min(to_apples) + min(from_apples_to_target)

def a_star_collect_apples(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], apples: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    open_list = PriorityQueue()
    close_set = set()
    support_list = {}

    collected = frozenset()
    h_val = heuristic_with_apples(start, apples, target)
    open_list.put((h_val, (start, 0, collected)))  # (f, (position, g, collected))
    support_list[(start, collected)] = 0
    parent = {(start, collected): None}

    while not open_list.empty():
        _, (current, current_cost, collected_apples) = open_list.get()
        state = (current, collected_apples)

        if state in close_set:
            continue
        close_set.add(state)

        # Collect apple if at one
        new_collected = set(collected_apples)
        if current in apples:
            new_collected.add(current)
        new_collected = frozenset(new_collected)

        # Goal condition
        if current == target and new_collected == apples:
            return build_path(parent, (current, new_collected))

        is_going_to_apple = new_collected != apples

        for neighbor in get_valid_moves(game_map, current, is_going_to_apple):
            neighbor_g = current_cost + 1
            neighbor_state = (neighbor, new_collected)
            if neighbor_state in support_list and neighbor_g >= support_list[neighbor_state]:
                continue

            support_list[neighbor_state] = neighbor_g
            parent[neighbor_state] = (current, collected_apples)
            h_val = heuristic_with_apples(neighbor, apples - new_collected, target)
            f_val = neighbor_g + h_val
            open_list.put((f_val, (neighbor, neighbor_g, new_collected)))

    print("Target node not reachable with all apples.")
    return None