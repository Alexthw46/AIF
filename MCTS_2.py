import random
from typing import Tuple, Set, List, Optional

import numpy as np

from utils import get_valid_moves, cached_bfs


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # (position, frozenset(collected_apples))
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.reward = 0.0


def is_terminal(state, target, apples):
    pos, collected = state
    return pos == target and collected == apples


def rollout_policy(game_map, state, target, apples, path_cache, heuristic=None):
    pos, collected = state
    collected = set(collected)
    visited = set()
    steps = 0
    max_steps = 150

    while steps < max_steps:
        if pos == target and collected == apples:
            break

        visited.add(pos)
        moves = get_valid_moves(game_map, pos, allow_diagonals=False)
        moves = [m for m in moves if m not in visited]  # avoid loops

        if not moves:
            break

        if heuristic is not None:
            # Prioritize:
            # 1. unexplored moves
            # 2. moves toward uncollected apples
            # 3. moves toward the target
            moves.sort(key=lambda m: (
                m in apples and m not in collected,
                heuristic(game_map, m, target, path_cache)
            ), reverse=True)
            pos = moves[0]
        else:
            pos = random.choice(moves)

        if pos in apples:
            collected.add(pos)
        steps += 1

    reward = len(collected) * 1.0
    if pos == target and collected == apples:
        reward += 5.0
    elif pos == target:
        reward += 1.0
    else:
        reward -= 2.0

    return reward


def expand(node, game_map, apples, visited_states):
    pos, collected = node.state
    collected = set(collected)

    for move in get_valid_moves(game_map, pos, allow_diagonals=False):
        new_collected = set(collected)
        if move in apples:
            new_collected.add(move)
        new_state = (move, frozenset(new_collected))
        if new_state not in visited_states:
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            visited_states.add(new_state)


def tree_policy(node, game_map, target, apples, C=1.4):
    # exploration constant
    while not is_terminal(node.state, target, apples):
        if node.children:
            # UCB1 selection
            node = max(
                node.children,
                key=lambda c: (c.reward / c.visits if c.visits > 0 else float("inf")) +
                              C * np.sqrt(np.log(node.visits + 1) / (c.visits + 1))
            )
        else:
            return node
    return node


def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent


def best_path(node):
    path = []
    while node:
        path.append(node.state[0])
        if node.children:
            node = max(node.children, key=lambda c: c.visits)
        else:
            break
    return path


def mcts(
        game_map: np.ndarray,
        start: Tuple[int, int],
        target: Tuple[int, int],
        apples: Set[Tuple[int, int]],
        iterations=1000,
        policy=rollout_policy,
        heuristic=None
) -> List[Tuple[int, int]]:
    root = MCTSNode((start, frozenset()))
    visited_states = {root.state}
    path_cache = {}
    for i in range(iterations):
        node = tree_policy(root, game_map, target, apples)
        if not is_terminal(node.state, target, apples):
            expand(node, game_map, apples, visited_states)
            if node.children:
                node = random.choice(node.children)
        reward = policy(game_map, node.state, target, apples, path_cache, heuristic=heuristic)
        backpropagate(node, reward)

        # Optional: early exit if we found a full path
        if is_terminal(node.state, target, apples):
            break

    return best_path(root)
