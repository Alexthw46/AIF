import heapq
import itertools
import random
from collections import defaultdict
from queue import PriorityQueue
from typing import Set

from utils import *


def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path


def bfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> list[tuple[int, int]] | None:
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


def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> list[tuple[
    int, int]] | None:
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
            # print("Target found!")
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


# ---------------------------------------------

def heuristic_with_apples_MST(current: Tuple[int, int], apples: Set[Tuple[int, int]], target: Tuple[int, int]) -> int:
    points = list(apples)
    if not points:
        return manhattan_distance(current, target)

    all_points = [current] + points + [target]
    edges = []

    # Build complete graph with manhattan distances
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            p1, p2 = all_points[i], all_points[j]
            dist = manhattan_distance(p1, p2)
            edges.append((dist, i, j))

    # Kruskal's algorithm to compute MST
    parent = list(range(len(all_points)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    mst_cost = 0
    edges.sort()
    for dist, i, j in edges:
        if union(i, j):
            mst_cost += dist

    return mst_cost


def a_star_collect_apples(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int],
                          apples: Set[Tuple[int, int]], weight: float = 1.0) -> \
        list[tuple[int, int]] | None:
    open_list = PriorityQueue()
    close_set = set()
    support_list = {}

    collected = frozenset()
    h_val = heuristic_with_apples_MST(start, apples, target)
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
            path = build_path(parent, (current, new_collected))
            # reformat path
            path = [state[0] for state in path]
            path = [(int(x), int(y)) for (x, y) in path]
            return path

        is_going_to_apple = new_collected != apples

        for neighbor in get_valid_moves(game_map, current, avoid_stairs=is_going_to_apple):
            neighbor_g = current_cost + 1
            neighbor_state = (neighbor, new_collected)
            if neighbor_state in support_list and neighbor_g >= support_list[neighbor_state]:
                continue

            support_list[neighbor_state] = neighbor_g
            parent[neighbor_state] = (current, collected_apples)
            h_val = heuristic_with_apples_MST(neighbor, apples - new_collected, target)
            f_val = neighbor_g + weight * h_val
            open_list.put((f_val, (neighbor, neighbor_g, new_collected)))

    print("Target node not reachable with all apples.")
    return None


def a_star_apple(
        game_map: np.ndarray,
        start: Tuple[int, int],
        target: Tuple[int, int],
        apple_positions: Set[Tuple[int, int]],
        heuristic: callable = manhattan_distance,
        apple_bonus: float = 0.75,
        weight: float = 1.0,
        **kwargs: dict
) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm that prioritizes paths close to apples ('%'), optional weighted A*.

    :param game_map: 2D grid representing the game state.
    :param start: Starting coordinate (x, y).
    :param target: Target coordinate (x, y) to reach.
    :param apple_positions: Set of coordinates where apples are located.
    :param heuristic: Heuristic function to estimate distance to target.
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
    f_start = heuristic(point1=start, point2=target, game_map=game_map)
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
                f = tentative_g + heuristic(point1=neighbor, point2=target, game_map=game_map) * weight
                open_list.put((f, neighbor))

    # No path found
    return []


def potential_field_path(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int],
                         apples: Set[Tuple[int, int]], max_steps=5000, heuristic: callable = manhattan_distance) -> \
        List[Tuple[int, int]] | None:
    def attractive_force(pos, goal, path_cache, weight=1.0):
        if heuristic == cached_bfs:
            return -weight * heuristic(game_map, pos, goal, path_cache)
        elif heuristic == manhattan_distance:
            return -weight * heuristic(pos, goal)

    def total_potential(pos, remaining_apples, target, path_cache):
        potential = attractive_force(pos, target, path_cache, weight=1)
        for apple in remaining_apples:
            potential += attractive_force(pos, apple, path_cache, weight=0.75)
        # Add small random noise to break ties/local minima
        potential += random.uniform(-0.3, 0.3)
        # Add repulsion from previous visits
        potential -= visit_count[pos] * 5
        return potential

    pos = start
    collected = set()
    path = [pos]
    steps = 0
    visit_count = defaultdict(int)
    path_cache = {}

    while steps < max_steps:
        steps += 1

        if pos in apples:
            collected.add(pos)

        if pos == target:
            return path

        remaining_apples = apples - collected
        visit_count[pos] += 1

        candidates = get_valid_moves(game_map, pos, allow_diagonals=False)
        if not candidates:
            break  # dead end

        best_move = max(candidates, key=lambda m: total_potential(m, remaining_apples, target, path_cache))

        if visit_count[best_move] > 10 or best_move == pos:
            print("being stuck")
            break  # likely stuck

        pos = best_move
        path.append(pos)

    print("Failed to reach target within potential field limits.")
    return None


def greedy_best_first_search(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int],
                             apples: Set[Tuple[int, int]], heuristic: callable = manhattan_distance) -> List[Tuple[
    int, int]] | None:
    def evaluate_heuristic(game_map, pos, collected, apples, target, path_cache):
        remaining_apples = apples - collected
        if not remaining_apples:
            # Se non ci sono mele da raccogliere, solo distanza dalla posizione al target
            if heuristic == manhattan_distance:
                return heuristic(pos, target)
            elif heuristic == cached_bfs:
                return heuristic(game_map, pos, target, path_cache)

        dist = 0
        current_pos = pos
        unvisited = set(remaining_apples)

        while unvisited:
            # Trova la mela più vicina alla posizione corrente
            if heuristic == manhattan_distance:
                next_apple = min(unvisited, key=lambda a: heuristic(current_pos, a))
                dist += heuristic(current_pos, next_apple)
                current_pos = next_apple
                unvisited.remove(next_apple)
            elif heuristic == cached_bfs:
                next_apple = min(unvisited, key=lambda a: heuristic(game_map, current_pos, a, path_cache))
                dist += heuristic(game_map, current_pos, next_apple, path_cache)
                current_pos = next_apple
                unvisited.remove(next_apple)

        # Aggiungi distanza dall'ultima mela al target
        if heuristic == manhattan_distance:
            dist += heuristic(current_pos, target)
        elif heuristic == cached_bfs:
            dist += heuristic(game_map, current_pos, target, path_cache)
        return dist

    start_state = (start, frozenset())  # position, collected_apples
    frontier = []
    path_cache = {}
    heapq.heappush(frontier,
                   (evaluate_heuristic(game_map, start, frozenset(), apples, target, path_cache), start_state, [start]))
    visited = set()

    while frontier:
        _, (pos, collected), path = heapq.heappop(frontier)
        if (pos, collected) in visited:
            continue
        visited.add((pos, collected))

        # Check if goal reached
        if pos == target and collected == apples:
            return path

        for move in get_valid_moves(game_map, pos, allow_diagonals=False):
            new_collected = set(collected)
            if move in apples:
                new_collected.add(move)
            new_state = (move, frozenset(new_collected))
            if new_state not in visited:
                new_path = path + [move]
                h = evaluate_heuristic(game_map, move, new_collected, apples, target, path_cache)
                heapq.heappush(frontier, (h, new_state, new_path))

    return None  # no path found


def beam_search_apple(game_map: np.ndarray,
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
        path = a_star_apple(game_map, a, b, heuristic=manhattan_distance,
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

        # Update 'best path' if any finished at target
        for net, _, _, pos, _, path in beam:
            if pos == target and net > best_net:
                best_net = net
                best_path = path

        # Only keep non-terminal entries in beam to allow further expansion
        beam = [entry for entry in beam if entry[3] != target]

    print(f"Best net reward: {best_net}, Path length: {len(best_path)}")
    return best_path
