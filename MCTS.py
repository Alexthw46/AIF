import random
from typing import Tuple, Set, List, Optional

import numpy as np

from utils import get_valid_moves


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # (position, collected_apples)
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.reward = 0.0


def is_terminal(state, target: Tuple[int, int], apples: Set[Tuple[int, int]]) -> bool:
    pos, collected = state
    return pos == target and collected == apples


def rollout_policy(game_map, state, target: Tuple[int, int], apples: Set[Tuple[int, int]], max_steps=150) -> float:
    # Random rollout policy, simulate until terminal or max steps
    pos, collected = state
    collected = set(collected)
    visited = set()
    steps = 0

    while steps < max_steps:
        if pos == target and collected == apples:
            break

        visited.add(pos)
        moves = get_valid_moves(game_map, pos)
        moves = [m for m in moves if m not in visited]  # avoid loops

        if not moves:
            break

        pos = random.choice(moves)

        if pos in apples:
            collected.add(pos)
        steps += 1

    # Reward for collected apples + bonus if target reached after collecting all
    reward = len(collected) * 1.0
    if pos == target and collected == apples:
        reward += 5  # bonus reward for full success
    elif pos == target:
        reward += 4
    else:  # penalization if you don't reach the downstairs target
        reward -= 2  # or set reward = 0
    return reward


def tree_policy(node, game_map, target: Tuple[int, int], apples: Set[Tuple[int, int]], C=1.4):
    """
    Traverse the tree from the given node using the UCB1 formula to select child nodes,
    until a leaf or terminal node is reached.

    :param node: The current MCTSNode to start selection from.
    :param game_map: The map of the environment.
    :param target: The target position as a tuple (row, col).
    :param apples: Set of positions of apples to collect.
    :param C: Exploration constant for UCB1.

    :return: The selected leaf or terminal MCTSNode.
    """
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


def expand(node, game_map, apples, visited_states):
    pos, collected = node.state
    collected = set(collected)

    for move in get_valid_moves(game_map, pos):
        new_collected = set(collected)
        if move in apples:
            new_collected.add(move)
        child_state = (move, frozenset(new_collected))
        new_state = (move, frozenset(new_collected))
        if new_state not in visited_states:
            child = MCTSNode(new_state, parent=node)
            node.children.append(child)
            visited_states.add(new_state)

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent


def best_path(node):
    path = []
    while node:
        path.append(node.state[0])  # position only
        if node.children:
            # Choose child with the highest visits
            node = max(node.children, key=lambda c: c.visits)
        else:
            node = None
    return path


def mcts(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], apples: Set[Tuple[int, int]],
         iterations=1000, C=1.4) -> List[Tuple[int, int]]:
    """
    Perform Monte Carlo Tree Search to find a path from start to target, collecting all apples.

    :param game_map: The nethack map as a numpy array.
    :param start: The starting position as a tuple (row, col).
    :param target: The target position as a tuple (row, col).
    :param apples: Set of positions of apples to collect.
    :param iterations: Number of MCTS iterations to perform.
    :param C: Exploration constant for UCB1.
    :return: List of positions representing the best path found.
    """
    root = MCTSNode((start, frozenset()))
    visited_states = {root.state}

    for _ in range(iterations):
        node = root
        # Selection
        while node.children:
            node = tree_policy(node, game_map, target, apples, C)
        # Expansion
        if not is_terminal(node.state, target, apples):
            expand(node, game_map, apples, visited_states)
            if node.children:
                node = random.choice(node.children)
        # Simulation
        reward = rollout_policy(game_map, node.state, target, apples)
        # Backpropagation
        backpropagate(node, reward)

    return best_path(root)
